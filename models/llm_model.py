"""
LLM model implementation for risky decision-making experiments.
"""

import os
import logging
import re
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
from datetime import datetime
from tqdm import tqdm

# Torch is only required for HuggingFace/vLLM paths. Allow OpenAI-only usage
# in environments where torch isn't installed.
try:
    import torch  # type: ignore
except Exception:  # noqa: BLE001 - allow import-less OpenAI runs
    torch = None  # type: ignore

# vLLM is only needed for HuggingFace inference and should not be imported in
# OpenAI-only runs (e.g., CPU Slurm jobs), because importing vLLM can trigger
# CUDA probing warnings/errors. We lazy-import vLLM inside `_init_huggingface`.
VLLM_AVAILABLE = False
LLM = None  # type: ignore
SamplingParams = None  # type: ignore

from utils.config import LLM_CONFIG
from utils.prompt_templates import LLM_PROMPTS, OPENAI_PROMPTS, HF_PROMPTS

# Setup logging
logger = logging.getLogger(__name__)

# Thread-local storage for OpenAI API calls
thread_local = threading.local()

class LLMModel:
    """LLM model for risky decision-making experiments."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        mode: str = "all",
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.95,
        max_num_batched_tokens: int = 8192,  # Increased for better throughput
        max_num_seqs: int = 4,  # Total batch size across all GPUs
        enforce_eager: bool = True,  # Always use eager mode for stability
        num_workers: int = 10,
        batch_size_per_device: int = 4,  # This will be the total batch size for vLLM
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM model.
        
        Args:
            model_name: Name of the model to use
            model_type: Type of model (huggingface or openai)
            mode: Experiment mode
                For exp1: base, cot, human, or all
                For exp2: exp2_within_individual, exp2_within_context, or exp2_all
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_num_batched_tokens: Maximum number of tokens to process in a batch
            max_num_seqs: Maximum number of sequences to process in a batch (total across all GPUs)
            enforce_eager: Whether to enforce eager mode
            num_workers: Number of worker threads for OpenAI API calls (default: 10)
            batch_size_per_device: Total batch size for processing (default: 4)
            api_key: OpenAI API key (required for OpenAI models)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.mode = mode
        self.num_workers = num_workers
        # For vLLM, this represents the total batch size, not per-device
        self.total_batch_size = batch_size_per_device  # Rename for clarity
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        
        # Initialize model based on type
        if model_type == "openai":
            self._init_openai(api_key)
        elif model_type == "huggingface":
            self._init_huggingface(
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI model."""
        if not api_key:
            raise ValueError("API key is required for OpenAI models")
        self.client = OpenAI(api_key=api_key)
    
    def _init_huggingface(
        self,
        tensor_parallel_size: Optional[int],
        gpu_memory_utilization: float,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        enforce_eager: bool
    ):
        """Initialize HuggingFace model."""
        # Lazy-import vLLM so OpenAI-only runs never import it.
        global VLLM_AVAILABLE, LLM, SamplingParams
        if not VLLM_AVAILABLE:
            try:
                from vllm import LLM as _VLLM_LLM, SamplingParams as _VLLM_SamplingParams  # type: ignore
                LLM = _VLLM_LLM  # type: ignore
                SamplingParams = _VLLM_SamplingParams  # type: ignore
                VLLM_AVAILABLE = True
            except Exception:
                VLLM_AVAILABLE = False

        # Ensure torch is imported and available
        if torch is None:
            raise ImportError(
                "torch is required for HuggingFace/vLLM model_type='huggingface', but it is not installed."
            )
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using CPU for model inference.")
        
        # Calculate available GPU memory and set device map
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            logger.info(f"Available GPUs: {num_gpus}, GPU Memory: {gpu_memory:.2f}GB")
            
            # Use all available GPUs for tensor parallelism (or specified amount)
            if tensor_parallel_size is not None:
                tensor_parallel_size = min(tensor_parallel_size, num_gpus)
                logger.info(f"Using specified tensor parallelism with {tensor_parallel_size} GPUs")
            else:
                tensor_parallel_size = num_gpus
                logger.info(f"Auto-detected tensor parallelism with {tensor_parallel_size} GPUs")
            
            # Set memory optimization flags
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_utilization)
        else:
            tensor_parallel_size = 1
        
        # Initialize vLLM if available (preferred for large models)
        if VLLM_AVAILABLE:
            try:
                # Step 1: Environment setup with debugging
                logger.info("🔧 Step 1: Setting up vLLM environment...")
                import tempfile
                import getpass
                username = getpass.getuser()
                cache_base = os.path.join(tempfile.gettempdir(), f"vllm_cache_{username}")
                os.makedirs(cache_base, exist_ok=True)
                logger.info(f"Cache directory: {cache_base}")
                
                # Essential environment variables for vLLM stability
                env_vars = {
                    "VLLM_CACHE_DIR": cache_base,
                    "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(torch.cuda.device_count())),
                    "NCCL_DEBUG": "WARN",  # Reduce NCCL logging
                    "NCCL_TIMEOUT": "1800",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "VLLM_LOGGING_LEVEL": "INFO",  # Reduce logging level
                    "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "1",  # Disable custom all-reduce to avoid P2P cache issues
                    "VLLM_USE_CUDA_GRAPH": "0",  # Disable CUDA graphs for stability
                    "VLLM_DO_NOT_TRACK": "1",  # Disable usage reporting to avoid disk quota issues
                    "HF_HUB_DOWNLOAD_TIMEOUT": "300",  # Increase HuggingFace download timeout
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable faster downloads
                    "TOKENIZERS_PARALLELISM": "false"  # Disable tokenizer parallelism warnings
                }
                
                # Set environment variables directly in the current process
                for key, value in env_vars.items():
                    os.environ[key] = value
                
                # Configure model parameters based on size
                model_size_gb = self._estimate_model_size(self.model_name)
                if model_size_gb > 100:  # For 70B+ models
                    max_model_len = 4096  # More conservative for large models
                    gpu_memory_utilization = 0.8  # Conservative for stability
                elif model_size_gb > 60:  # For 32B+ models
                    max_model_len = 4096
                    gpu_memory_utilization = 0.8
                else:
                    max_model_len = 4096
                    gpu_memory_utilization = 0.8

                # Initialize vLLM with environment variables
                init_params = {
                    "model": self.model_name,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "max_num_seqs": max_num_seqs,
                    "max_model_len": max_model_len,
                    "trust_remote_code": True,
                    "dtype": "bfloat16",
                    "enforce_eager": True,
                    "disable_custom_all_reduce": True,  # Disable custom all-reduce to avoid cache issues
                    "max_num_batched_tokens": 8192,  # Increased for better throughput
                    "swap_space": 4  # GB of CPU swap space per GPU
                }
                
                logger.info("Initialization parameters:")
                for key, value in init_params.items():
                    logger.info(f"  {key}: {value}")
                
                logger.info("🔄 Starting vLLM initialization... (this may take several minutes)")
                logger.info("If it hangs here, the issue is during worker startup or model loading")
                
                # Add progress monitoring
                import time
                start_time = time.time()
                
                try:
                    self.vllm_model = LLM(**init_params)
                    init_time = time.time() - start_time
                    logger.info(f"✅ vLLM initialized successfully in {init_time:.1f} seconds!")
                    
                except Exception as init_e:
                    init_time = time.time() - start_time
                    logger.error(f"❌ vLLM initialization failed after {init_time:.1f} seconds")
                    logger.error(f"Failure type: {type(init_e).__name__}")
                    logger.error(f"Failure message: {str(init_e)}")
                    
                    # Check if workers are still running
                    import psutil
                    vllm_processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if 'vllm' in proc.info['name'].lower() or any('vllm' in arg for arg in proc.info['cmdline']):
                                vllm_processes.append(proc.info)
                        except:
                            pass
                    
                    logger.error(f"Found {len(vllm_processes)} vLLM-related processes")
                    for proc in vllm_processes:
                        logger.error(f"  PID {proc['pid']}: {proc['name']}")
                    
                    raise init_e
                
                # Step 6: Test generation
                logger.info("🧪 Step 6: Testing generation...")
                try:
                    from vllm import SamplingParams
                    test_params = SamplingParams(max_tokens=5, temperature=0.0)
                    logger.info("Running test generation: 'Hello'")
                    test_output = self.vllm_model.generate(["Hello"], test_params)
                    logger.info(f"✅ Test successful! Output: {test_output[0].outputs[0].text}")
                except Exception as test_e:
                    logger.error(f"❌ Test generation failed: {test_e}")
                    raise test_e
                
                # Step 7: Finalization
                logger.info("🎉 Step 7: vLLM setup complete!")
                
                # Initialize tokenizer for prompt formatting (needed even with vLLM)
                logger.info("Initializing tokenizer for prompt formatting...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Deallocate HuggingFace model to save memory (but keep tokenizer)
                # We'll use vLLM's logprobs capability for probability calculations
                self.model = None
                logger.info("HuggingFace model deallocated to save memory (using vLLM for both generation and probabilities)")
                
            except Exception as e:
                logger.error("="*60)
                logger.error("🚨 vLLM INITIALIZATION FAILED")
                logger.error("="*60)
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                
                import traceback
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                
                # Diagnostic information
                logger.error("\n📊 DIAGNOSTIC INFORMATION:")
                logger.error(f"Model: {self.model_name}")
                logger.error(f"Tensor parallel size: {tensor_parallel_size}")
                logger.error(f"Available GPUs: {torch.cuda.device_count()}")
                logger.error(f"CUDA version: {torch.version.cuda}")
                
                try:
                    import vllm
                    logger.error(f"vLLM version: {vllm.__version__}")
                except:
                    logger.error("vLLM version: unknown")
                
                # Check for zombie processes
                import psutil
                zombie_count = 0
                for proc in psutil.process_iter():
                    try:
                        if proc.status() == psutil.STATUS_ZOMBIE:
                            zombie_count += 1
                    except:
                        pass
                logger.error(f"Zombie processes: {zombie_count}")
                
                # Cleanup
                if hasattr(self, 'vllm_model'):
                    try:
                        del self.vllm_model
                    except:
                        pass
                
                self.vllm_model = None
                logger.error("❌ vLLM failed - stopping execution as requested")
                raise e
        else:
            logger.error("vLLM not available!")
            raise ImportError("vLLM is required but not available")
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB based on model name."""
        name_lower = model_name.lower()
        if "72b" in name_lower or "70b" in name_lower:
            return 140.0  # ~140GB for 70B/72B models in bfloat16
        elif "32b" in name_lower:
            return 64.0   # ~64GB for 32B models in bfloat16
        elif "30b" in name_lower:
            return 60.0
        elif "14b" in name_lower:
            return 28.0   # ~28GB for 14B models in bfloat16
        elif "13b" in name_lower:
            return 26.0
        elif "8b" in name_lower:
            return 16.0   # ~16GB for 8B models in bfloat16
        elif "7b" in name_lower:
            return 14.0
        else:
            return 14.0  # Default assumption
    

    
    def _init_huggingface_fallback(self, tensor_parallel_size: int, gpu_memory_utilization: float):
        """Initialize standard HuggingFace model as fallback."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            device_map = "auto"
            max_memory = {i: f"{int(gpu_memory * gpu_memory_utilization)}GiB" for i in range(num_gpus)}
        else:
            device_map = "cpu"
            max_memory = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model for generation and probability probing
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            load_in_8bit=False
        )
    
    def _format_prompt(
        self,
        mode: str,
        probabilities: List[float],
        outcomes: List[float],
        think_aloud: Optional[str] = None,
        question_context: Optional[str] = None,
        examples: Optional[str] = None,
        example_type: str = "think_aloud"
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Format prompt for the specified mode.
        
        Args:
            mode: Experiment mode
                For exp1: base, cot, human, or all
                For exp2: base_within_individual, cot_within_individual, base_within_context, cot_within_context
            probabilities: List of probabilities [p1, p2, p3, p4]
            outcomes: List of outcomes [o1, o2, o3, o4]
            think_aloud: Human think-aloud text (for human mode)
            question_context: Question context
            examples: Formatted examples for in-context learning
            example_type: Type of information in examples ("think_aloud", "choice", "both")
            
        Returns:
            Formatted prompt (string for HuggingFace, list of dicts for OpenAI)
        """
        # Get appropriate content type text based on example_type
        def get_example_content_type(example_type: str) -> str:
            if example_type == "think_aloud":
                return "thoughts"
            elif example_type == "choice":
                return "decisions"
            elif example_type == "both":
                return "thoughts and decisions"
            else:
                return "thoughts"  # Default fallback
        
        example_content_type = get_example_content_type(example_type)
        
        # Select appropriate prompt template based on model type and mode
        if self.model_type == "openai":
            # For OpenAI models, use the appropriate template based on mode
            if mode in ["base_within_individual", "cot_within_individual", "base_within_context", "cot_within_context"]:
                # In-context learning modes (exp2)
                prompt_template = OPENAI_PROMPTS[mode]
            elif mode in ["base", "cot", "human"]:
                # Zero-shot learning modes (exp1)
                prompt_template = OPENAI_PROMPTS[mode]
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        else:
            # For HuggingFace models
            if mode in ["base_within_individual", "cot_within_individual", "base_within_context", "cot_within_context"]:
                # In-context learning modes (exp2)
                prompt_template = HF_PROMPTS[mode]
            elif mode in ["base", "cot", "human"]:
                # Zero-shot learning modes (exp1)
                prompt_template = HF_PROMPTS[mode]
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        
        # Format messages with proper content
        messages = []
        for msg in prompt_template:
            # Format content with placeholders
            content = msg["content"].format(
                question_context=question_context if question_context else "",
                think_aloud=think_aloud if think_aloud else "",
                examples=examples if examples else "",
                example_content_type=example_content_type
            )
            messages.append({"role": msg["role"], "content": content})
        
        if self.model_type == "openai":
            # For OpenAI API, return the messages directly
            return messages
        else:
            # For HuggingFace, use apply_chat_template to format the messages
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    continue_final_message=True
                )
            else:
                raise ValueError(f"Model {self.model_name} does not support chat template. Please use a model that supports chat templates.")
    
    def _get_max_tokens(self, mode: str) -> int:
        """
        Get max tokens for generation based on mode.
        
        Args:
            mode: Experiment mode (base, cot, human)
            
        Returns:
            Maximum number of tokens to generate
        """
        return 2048  # Use 2048 tokens for all modes to allow for longer responses
    
    def _extract_choice(self, response: str) -> Optional[int]:
        """
        Extract choice from response by looking for the choice pattern.
        
        Args:
            response: Model response
            
        Returns:
            0 for Option A, 1 for Option B, None if no choice found
        """
        # Clean up the response
        response = response.strip().lower()
        
        # Look for the full phrase first
        if "i will choose option a" in response:
            return 0
        elif "i will choose option b" in response:
            return 1
        
        # If the response contains "I will choose Option" followed by just the letter
        if "i will choose option" in response:
            # Find what comes after "i will choose option"
            choice_start = response.find("i will choose option") + len("i will choose option")
            remaining = response[choice_start:].strip()
            
            # Check if it starts with A or B
            if remaining.startswith('a'):
                return 0
            elif remaining.startswith('b'):
                return 1
        
        # Fallback: look for just "option a" or "option b" 
        if "option a" in response and "option b" not in response:
            return 0
        elif "option b" in response and "option a" not in response:
            return 1
        
        # Last resort: look for just "a." or "b." at the end
        if response.endswith('a.') or response.endswith(' a.'):
            return 0
        elif response.endswith('b.') or response.endswith(' b.'):
            return 1
        
        # Look for the last occurrence of "I will choose" in the response
        last_choice = response.rfind("i will choose")
        if last_choice != -1:
            # Get everything after the last "I will choose"
            remaining = response[last_choice:].strip()
            if "option a" in remaining or "a." in remaining:
                return 0
            elif "option b" in remaining or "b." in remaining:
                return 1
        
        # Look for the last occurrence of "I choose" in the response
        last_choice = response.rfind("i choose")
        if last_choice != -1:
            # Get everything after the last "I choose"
            remaining = response[last_choice:].strip()
            if "option a" in remaining or "a." in remaining:
                return 0
            elif "option b" in remaining or "b." in remaining:
                return 1
        
        # Look for the last occurrence of "I would choose" in the response
        last_choice = response.rfind("i would choose")
        if last_choice != -1:
            # Get everything after the last "I would choose"
            remaining = response[last_choice:].strip()
            if "option a" in remaining or "a." in remaining:
                return 0
            elif "option b" in remaining or "b." in remaining:
                return 1
        
        # Look for the last occurrence of "I will select" in the response
        last_choice = response.rfind("i will select")
        if last_choice != -1:
            # Get everything after the last "I will select"
            remaining = response[last_choice:].strip()
            if "option a" in remaining or "a." in remaining:
                return 0
            elif "option b" in remaining or "b." in remaining:
                return 1
        
        return None
    
    def _generate_single_openai(
        self,
        prompt: List[Dict[str, str]],
        max_tokens: int
    ) -> Tuple[str, List[Dict[str, str]], object]:
        """Generate response for a single prompt using OpenAI API with logprobs."""
        try:
            # Get both generation and logprobs in a single call
            # Some newer OpenAI chat models use `max_completion_tokens`
            # instead of `max_tokens`.
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_completion_tokens=max_tokens,
                    temperature=0.7,
                    logprobs=True,
                    top_logprobs=20,  # Get top 20 alternative tokens for each position
                )
            except Exception as e:
                # Some OpenAI models do not allow requesting logprobs at all.
                msg = str(e)
                if "not allowed to request logprobs" in msg.lower():
                    # Retry without logprobs; downstream we will fall back to
                    # uniform choice probabilities since we only need the choice.
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prompt,
                            max_completion_tokens=max_tokens,
                            temperature=0.7,
                        )
                        api_response = None
                    except Exception:
                        # Last-resort fallback with max_tokens for older API surface
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prompt,
                            max_tokens=max_tokens,
                            temperature=0.7,
                        )
                        api_response = None
                elif "max_tokens" in msg or "Unsupported parameter" in msg:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        logprobs=True,
                        top_logprobs=20,  # Get top 20 alternative tokens for each position
                    )
                    api_response = response
                else:
                    raise
            else:
                api_response = response
            
            # Extract the response content
            if not response or not hasattr(response, 'choices') or not response.choices:
                return "", prompt, None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                return "", prompt, None
                
            response_content = choice.message.content
            if response_content is None:
                response_content = ""
            else:
                response_content = response_content.strip()
            
            # Return response content and the API response (if we have logprobs).
            return response_content, prompt, api_response
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return "", prompt, None
    
    def _generate_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        mode: str
    ) -> List[Tuple[str, Optional[List[Dict[str, str]]], object]]:
        """
        Generate responses in batch.
        
        Args:
            prompts: List of prompts
            mode: Experiment mode
            
        Returns:
            List of tuples containing (response, processed_prompt, api_response)
        """
        max_tokens = self._get_max_tokens(mode)
        
        if self.model_type == "openai":
            # OpenAI batch generation with multi-threading while maintaining order
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks with indices
                futures = [
                    executor.submit(self._generate_single_openai, prompt, max_tokens)
                    for prompt in prompts
                ]
                
                # Collect results in submission order with progress bar
                responses = []
                with tqdm(total=len(futures), desc="OpenAI API calls", leave=False) as pbar:
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                            if result is None:
                                responses.append(("", [], None))
                            else:
                                response, processed_prompt, api_response = result
                                if response is None:
                                    response = ""  # Convert None to empty string
                                responses.append((response, processed_prompt, api_response))
                        except Exception as e:
                            logger.error(f"Error processing prompt {i}: {str(e)}")
                            responses.append(("", [], None))
                        pbar.update(1)
            
            return responses
        else:
            # HuggingFace generation
            if VLLM_AVAILABLE and self.vllm_model is not None:
                # Use vLLM for generation with logprobs - process in batches of total_batch_size
                responses = []
                # Use the configured total batch size (e.g., 4) instead of per-device size
                batch_size = min(self.total_batch_size, self.max_num_seqs)
                num_batches = (len(prompts) + batch_size - 1) // batch_size
                
                logger.info(f"Processing {len(prompts)} prompts in {num_batches} batches of size {batch_size}")
                
                with tqdm(total=num_batches, desc="vLLM generation", leave=False) as pbar:
                    for i in range(0, len(prompts), batch_size):
                        batch_prompts = prompts[i:i + batch_size]
                        
                        try:
                            # Validate prompt types
                            for j, prompt in enumerate(batch_prompts):
                                if not isinstance(prompt, str):
                                    logger.error(f"Non-string prompt detected at index {j}: {type(prompt)}")
                            
                            sampling_params = SamplingParams(
                                max_tokens=max_tokens,
                                temperature=0.7,
                                top_p=LLM_CONFIG["top_p"],
                                n=1,  # Generate one response per prompt
                                stop=None,  # No stop tokens
                                skip_special_tokens=True,
                                logprobs=20  # Get top 20 logprobs for probability extraction
                            )
                            
                            batch_outputs = self.vllm_model.generate(batch_prompts, sampling_params)
                            batch_responses = []
                            
                            for output in batch_outputs:
                                if output.outputs and len(output.outputs) > 0:
                                    response_text = output.outputs[0].text.strip()
                                    # Store the entire output object for probability extraction
                                    batch_responses.append((response_text, None, output))
                                else:
                                    batch_responses.append(("", None, None))
                            
                            responses.extend(batch_responses)
                            
                        except Exception as e:
                            logger.error(f"Error in vLLM batch generation: {e}")
                            # Add error responses for this batch
                            for _ in batch_prompts:
                                responses.append(("", None, None))
                        
                        pbar.update(1)
                        
            else:
                # Use standard HuggingFace inference - process in smaller batches
                responses = []
                # Use smaller batches for HF to avoid memory issues
                hf_batch_size = min(2, self.total_batch_size)  
                num_batches = (len(prompts) + hf_batch_size - 1) // hf_batch_size
                
                with tqdm(total=num_batches, desc="HuggingFace generation", leave=False) as pbar:
                    for i in range(0, len(prompts), hf_batch_size):
                        batch_prompts = prompts[i:i + hf_batch_size]
                        
                        try:
                            # Tokenize batch
                            inputs = self.tokenizer(
                                batch_prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=4096  # Limit input length
                            )
                            if torch.cuda.is_available():
                                inputs = {k: v.cuda() for k, v in inputs.items()}
                            
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=0.7,
                                    top_p=LLM_CONFIG["top_p"],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    do_sample=True,
                                    eos_token_id=self.tokenizer.eos_token_id
                                )
                                
                                                                # Decode only the new tokens (completion) for each sequence in batch
                                input_lengths = inputs["input_ids"].shape[1]
                                for j, output_seq in enumerate(outputs):
                                    new_tokens = output_seq[input_lengths:]
                                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                    response = response.strip()
                                    responses.append((response, None, None))
                                    
                        except Exception as e:
                            logger.error(f"Error in HuggingFace batch generation: {e}")
                            # Add error responses for this batch
                            for _ in batch_prompts:
                                responses.append(("", None, None))
                        
                        pbar.update(1)
        
        return responses
    
    def _process_response(self, response: Union[str, Tuple[str, Optional[List[Dict[str, str]]]]], mode: str) -> str:
        """
        Process the model's response based on the mode.
        
        Args:
            response: Raw model response (string or tuple of (response, processed_prompt))
            mode: Experiment mode (base, cot, human, or in-context modes)
            
        Returns:
            Processed response with system/user parts removed but assistant part kept
        """
        # Handle tuple response (from _generate_batch)
        if isinstance(response, tuple):
            response = response[0]  # Extract just the response string
        
        # Clean up any remaining formatting artifacts
        response = response.strip()
        
        # Remove any system/user role labels that might appear in the generated content
        # but keep assistant content
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are system or user role labels
            if line.lower() in ['user:', 'system:', '<|im_start|>user', '<|im_start|>system', '<|im_end|>']:
                continue
            # Remove system/user prefixes from lines, but keep assistant content
            if line.lower().startswith('user:'):
                continue  # Skip user lines entirely
            elif line.lower().startswith('system:'):
                continue  # Skip system lines entirely
            elif line.lower().startswith('assistant:'):
                # Keep assistant content but remove the prefix
                line = line[10:].strip()
            
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining special tokens except for assistant content
        response = response.replace('<|im_start|>user', '').replace('<|im_start|>system', '').replace('<|im_end|>', '').strip()
        
        # For both exp1 and exp2, we want to ensure the response ends with a clear choice
        # Look for the last occurrence of "I will choose" in the response
        last_choice = response.lower().rfind("i will choose")
        if last_choice != -1:
            # Get everything after the last "I will choose"
            remaining = response[last_choice:].strip()
            # If it doesn't already start with "I will choose Option", add it
            if not remaining.lower().startswith("i will choose option"):
                response = "I will choose Option " + remaining
        else:
            # If no "I will choose" found, ensure the response starts with it
            if not response.lower().startswith("i will choose option"):
                response = "I will choose Option " + response
        
        return response
    
    def predict(
        self,
        probabilities: List[float],
        outcomes: List[float],
        think_aloud: Optional[str] = None,
        question_context: Optional[str] = None,
        mode: Optional[str] = None,
        examples: Optional[str] = None,
        example_type: str = "think_aloud"
    ) -> Dict:
        """
        Make prediction(s) for the given input.
        
        Args:
            probabilities: List of probabilities [p1, p2, p3, p4]
            outcomes: List of outcomes [o1, o2, o3, o4]
            think_aloud: Human think-aloud text (for human mode)
            question_context: Question context
            mode: Override the default mode
            examples: Formatted examples for in-context learning
            example_type: Type of information in examples ("think_aloud", "choice", "both")
            
        Returns:
            Dictionary containing generation, choice, probabilities, and processed prompt
        """
        if mode is None:
            mode = self.mode
        
        # Set current mode for probability extraction
        self._current_mode = mode
        
        if mode == "all":
            # Run all modes
            results = {}
            for m in ["base", "cot", "human"]:
                results[m] = self.predict(
                    probabilities,
                    outcomes,
                    think_aloud,
                    question_context,
                    m,
                    examples,
                    example_type
                )
            return results
        
        # Format prompt
        prompt = self._format_prompt(
            mode,
            probabilities,
            outcomes,
            think_aloud,
            question_context,
            examples,
            example_type
        )
        
        # Generate response
        if self.model_type == "openai":
            # For single predictions, use _generate_single_openai directly
            response, processed_prompt, api_response = self._generate_single_openai(prompt, self._get_max_tokens(mode))
        else:
            response_tuple = self._generate_batch([prompt], mode)[0]
            response = response_tuple[0]  # Extract just the response string
            processed_prompt = None
            api_response = None
        
        # Process response
        processed_response = self._process_response(response, mode)
        
        # Extract choice
        choice = self._extract_choice(processed_response)
        
        # Get choice probabilities
        prob_A, prob_B = self._get_choice_probabilities(prompt, processed_response, api_response)
        
        result = {
            "generation": processed_response,
            "choice": choice,
            "probabilities": {
                "A": prob_A,
                "B": prob_B
            }
        }
        
        # Add processed prompt for OpenAI models
        if self.model_type == "openai":
            result["processed_prompt"] = processed_prompt
        
        return result

    def _get_choice_probabilities(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        response: str,
        api_response: object = None
    ) -> Tuple[float, float]:
        """
        Get choice probabilities from model.
        
        Args:
            prompt: Input prompt
            response: Model response text
            api_response: API response object (for OpenAI models)
            
        Returns:
            Tuple of (prob_A, prob_B)
        """
        if self.model_type == "openai":
            # For OpenAI models, some variants do not support logprobs. When
            # api_response is None (e.g., we had to call without logprobs),
            # we skip probability extraction and return sentinel values so that
            # downstream analysis can treat them as "probabilities unavailable".
            if api_response is None:
                logger.info(
                    "OpenAI logprobs unavailable for this model; returning "
                    "sentinel choice probabilities (-1.0, -1.0)."
                )
                return -1.0, -1.0
            # Otherwise, use the logprobs already returned by the generation call.
            return self._extract_choice_probabilities_from_logprobs(api_response)
            
        else:
            # For HuggingFace models, we need to find where the choice decision is made
            # and get the probabilities at that point
            
            # Ensure prompt is a string for HuggingFace models
            if not isinstance(prompt, str):
                logger.error(f"Expected string prompt for HuggingFace model, got {type(prompt)}")
                return 0.5, 0.5
            
            # Check if we have a HuggingFace model available (not deallocated for vLLM)
            if self.model is None and hasattr(self, 'vllm_model') and self.vllm_model is not None:
                # Use vLLM for probability calculation
                # Need to extract mode - this is a bit of a hack, but we can infer from prompt structure
                inferred_mode = "base"
                if hasattr(self, '_current_mode'):
                    inferred_mode = self._current_mode
                prob_A, prob_B = self._get_choice_probabilities_vllm(prompt, response, api_response, inferred_mode)
            elif self.model is not None:
                # Use standard HuggingFace model
                # Build the exact context for probability extraction
                last_decision_start = response.rfind("I will choose Option")
                if last_decision_start != -1:
                    # Include ALL generated content up to and including "I will choose Option "
                    response_up_to_decision = response[:last_decision_start + len("I will choose Option ")]
                    full_text = prompt + response_up_to_decision
                else:
                    # If no decision point found, append the decision phrase manually
                    full_text = prompt + " I will choose Option "
                
                try:
                    # Tokenize the full text
                    inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits[0, -1, :]  # Get logits at the last position
                        
                        # Always probe " A" and " B" tokens (with space prefix)
                        token_A = None
                        token_B = None
                        
                        try:
                            # Encode " A" and " B" tokens with space prefix
                            encoded_A = self.tokenizer.encode(" A", add_special_tokens=False)
                            encoded_B = self.tokenizer.encode(" B", add_special_tokens=False)
                            
                            if len(encoded_A) > 0 and len(encoded_B) > 0:
                                token_A = encoded_A[0]
                                token_B = encoded_B[0]
                                logger.debug(f"Using space-prefixed tokens: ' A' -> {token_A}, ' B' -> {token_B}")
                            else:
                                logger.error("Empty token encoding for ' A' and ' B'")
                                prob_A, prob_B = 0.5, 0.5
                                return prob_A, prob_B
                                
                        except (IndexError, KeyError, Exception) as e:
                            logger.error(f"Failed to encode ' A' and ' B' tokens: {e}")
                            prob_A, prob_B = 0.5, 0.5
                            return prob_A, prob_B
                        
                        if token_A is not None and token_B is not None:
                            # Get logits for both tokens
                            logit_A = logits[token_A].item()
                            logit_B = logits[token_B].item()
                            
                            logger.debug(f"HuggingFace logits - A: {logit_A:.6f}, B: {logit_B:.6f}")
                            
                            # Convert to probabilities
                            import torch
                            probs = torch.softmax(torch.tensor([logit_A, logit_B]), dim=0)
                            prob_A, prob_B = probs.tolist()
                            
                            logger.debug(f"HuggingFace probabilities - A: {prob_A:.6f}, B: {prob_B:.6f}")
                        else:
                            # This should not happen with our current logic
                            logger.error("Unexpected state: token_A or token_B is None after successful encoding")
                            prob_A, prob_B = 0.5, 0.5
                            
                except Exception as e:
                    logger.error(f"Error in HuggingFace probability calculation: {e}")
                    prob_A, prob_B = 0.5, 0.5
            else:
                # No model available for probability calculation
                logger.warning("No model available for probability calculation - returning uniform probabilities")
                prob_A, prob_B = 0.5, 0.5
        
        return prob_A, prob_B
    
    def _get_choice_probabilities_vllm(self, prompt: str, response: str, vllm_output: object = None, mode: str = "base") -> Tuple[float, float]:
        """
        Get choice probabilities directly from vLLM generation logprobs.
        No additional forward pass needed - uses the logprobs from the original generation.
        
        Args:
            prompt: Input prompt
            response: Model response
            vllm_output: vLLM output object containing logprobs
            mode: Experiment mode (base, cot, human)
            
        Returns:
            Tuple of (prob_A, prob_B)
        """
        try:
            if vllm_output is None:
                return 0.5, 0.5
            
            if not hasattr(vllm_output, 'outputs') or not vllm_output.outputs:
                return 0.5, 0.5
            
            output = vllm_output.outputs[0]
            if not hasattr(output, 'logprobs') or not output.logprobs:
                return 0.5, 0.5
            
            # Different logic based on mode
            if mode == "base" or mode == "human":
                # Base/Human mode: Model should generate A or B, but need to find the right position
                if len(output.logprobs) == 0:
                    return 0.5, 0.5
                
                # Find the position where A or B was generated
                generated_text = output.text
                choice_position = self._find_choice_token_position(output, generated_text)
                
                if choice_position == -1:
                    return 0.5, 0.5
                
                last_token_logprobs = output.logprobs[choice_position]
                
            elif mode == "cot":
                # CoT mode: Find where "I will choose Option A/B" occurs in the generated sequence
                choice_position = self._find_choice_position_in_generation(output, response)
                
                if choice_position == -1:
                    return 0.5, 0.5
                
                if choice_position >= len(output.logprobs):
                    return 0.5, 0.5
                
                last_token_logprobs = output.logprobs[choice_position]
            
            # Convert token IDs to strings and extract A/B probabilities
            token_str_logprobs = {}
            for token_id, logprob_obj in last_token_logprobs.items():
                logprob = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
                try:
                    if isinstance(token_id, int):
                        token_str = self.tokenizer.decode([token_id])
                    else:
                        token_str = str(token_id)
                    token_str_logprobs[token_str] = logprob
                except Exception:
                    continue
            
            # Look for A and B tokens
            prob_A = None
            prob_B = None
            
            for token_str, logprob in token_str_logprobs.items():
                # Primary: space-prefixed tokens
                if token_str == " A":
                    prob_A = logprob
                elif token_str == " B":
                    prob_B = logprob
                # Alternative: bare tokens (fallback)
                elif token_str.strip().upper() == "A" and prob_A is None:
                    prob_A = logprob
                elif token_str.strip().upper() == "B" and prob_B is None:
                    prob_B = logprob
            
            # If we found both tokens, use them
            if prob_A is not None and prob_B is not None:
                # Convert log probabilities to probabilities and normalize
                import torch
                logits = torch.tensor([prob_A, prob_B])
                probs = torch.softmax(logits, dim=0)
                prob_A_norm, prob_B_norm = probs.tolist()
                
                return prob_A_norm, prob_B_norm
            
            # If we found only one token, try alternative extraction methods
            elif prob_A is not None or prob_B is not None:
                # Try to find the missing token in alternatives or estimate
                if prob_A is None:
                    # Look for alternative representations of A in the converted strings
                    for token_str in token_str_logprobs:
                        if token_str.strip().upper() == "A":
                            prob_A = token_str_logprobs[token_str]
                            break
                    
                    # If still not found, estimate as much lower probability
                    if prob_A is None:
                        prob_A = prob_B - 5.0  # Much lower logprob
                
                if prob_B is None:
                    # Look for alternative representations of B in the converted strings
                    for token_str in token_str_logprobs:
                        if token_str.strip().upper() == "B":
                            prob_B = token_str_logprobs[token_str]
                            break
                    
                    # If still not found, estimate as much lower probability
                    if prob_B is None:
                        prob_B = prob_A - 5.0  # Much lower logprob
                
                # Normalize the probabilities
                import torch
                logits = torch.tensor([prob_A, prob_B])
                probs = torch.softmax(logits, dim=0)
                prob_A_norm, prob_B_norm = probs.tolist()
                
                return prob_A_norm, prob_B_norm
            
            # If we didn't find either token, check if the model is generating something else
            else:
                # Try to extract from the actual generated token if it's A or B
                generated_token = output.text.strip()
                if generated_token.upper() in ["A", "B"]:
                    if generated_token.upper() == "A":
                        # Model strongly prefers A
                        return 0.9, 0.1
                    else:
                        # Model strongly prefers B
                        return 0.1, 0.9
            
            # Fallback if no probabilities could be extracted
            return 0.5, 0.5
            
        except Exception as e:
            logger.error(f"Error getting choice probabilities from vLLM: {e}")
            return 0.5, 0.5
    
    def _find_choice_position_in_generation(self, output, response: str) -> int:
        """
        Find the token position where the choice (A or B) was made in CoT generation.
        
        Args:
            output: vLLM output object
            response: Generated response text
            
        Returns:
            Token position index, or -1 if not found
        """
        try:
            # Find "I will choose Option" followed by A or B in the response
            choice_start = response.rfind("I will choose Option")
            if choice_start == -1:
                return -1
            
            # Find what comes after "I will choose Option"
            choice_phrase_end = choice_start + len("I will choose Option")
            remaining_text = response[choice_phrase_end:].strip()
            
            if not remaining_text or remaining_text[0].upper() not in ["A", "B"]:
                return -1
            
            # Now we need to find this position in the tokenized output
            # This is approximate - we'll look for tokens near the end that are A or B
            for i in range(len(output.logprobs) - 1, max(-1, len(output.logprobs) - 10), -1):
                # Convert tokens at this position to see if any are A or B
                token_logprobs = output.logprobs[i]
                for token_id, logprob_obj in token_logprobs.items():
                    try:
                        if isinstance(token_id, int):
                            token_str = self.tokenizer.decode([token_id])
                        else:
                            token_str = str(token_id)
                        
                        if token_str.strip().upper() in ["A", "B"]:
                            return i
                    except:
                        continue
            
            # If not found, use last position as fallback
            return len(output.logprobs) - 1
            
        except Exception as e:
            logger.error(f"Error finding choice position: {e}")
            return -1
    
    def _find_choice_token_position(self, output, generated_text: str) -> int:
        """
        Find the token position where A or B was generated in base/human modes.
        
        Args:
            output: vLLM output object
            generated_text: The generated text
            
        Returns:
            Token position index, or -1 if not found
        """
        try:
            # Look for A or B in the generated text
            choice_char = None
            if "A" in generated_text:
                choice_char = "A"
            elif "B" in generated_text:
                choice_char = "B"
            
            if not choice_char:
                return -1
            
            # Search through logprob positions to find where this choice was generated
            # Look at the last few positions since choice is usually near the end
            search_positions = min(len(output.logprobs), 5)  # Check last 5 positions
            
            for i in range(len(output.logprobs) - search_positions, len(output.logprobs)):
                if i < 0:
                    continue
                    
                token_logprobs = output.logprobs[i]
                
                # Convert token IDs to strings and check if any match our choice
                for token_id, logprob_obj in token_logprobs.items():
                    try:
                        if isinstance(token_id, int):
                            token_str = self.tokenizer.decode([token_id])
                        else:
                            token_str = str(token_id)
                        
                        # Check if this token contains our choice character (be flexible with formats)
                        cleaned_token = token_str.strip().upper()
                        if (cleaned_token == choice_char or 
                            cleaned_token == f"{choice_char}." or
                            token_str == f" {choice_char}" or 
                            token_str == f" {choice_char}." or
                            token_str == f"{choice_char}." or
                            token_str.strip() == f"{choice_char}." or
                            # Also check if the raw token matches
                            token_str.upper().strip('.').strip() == choice_char):
                            return i
                            
                    except Exception:
                        continue
            
            # Fallback: use second-to-last position (before EOS token)
            return max(0, len(output.logprobs) - 2)
            
        except Exception as e:
            logger.error(f"Error finding choice token position: {e}")
            return -1
    
    def _extract_choice_probabilities_from_logprobs(self, api_response) -> Tuple[float, float]:
        """
        Extract choice probabilities from OpenAI API response logprobs.
        Looks for A and B tokens in the generated content.
        
        Args:
            api_response: OpenAI API response object with logprobs
            
        Returns:
            Tuple of (prob_A, prob_B)
        """
        try:
            if not api_response or not hasattr(api_response, 'choices') or not api_response.choices:
                return 0.5, 0.5
                
            choice = api_response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs:
                return 0.5, 0.5
                
            logprobs = choice.logprobs
            if not hasattr(logprobs, 'content') or not logprobs.content:
                return 0.5, 0.5
            
            # Find the decision tokens by looking through all token positions
            prob_A = -float("inf")
            prob_B = -float("inf")
            found_decision = False
            
            # Search through all token positions (from last to first)
            for token_idx in range(len(logprobs.content) - 1, -1, -1):
                token_info = logprobs.content[token_idx]
                if not token_info.top_logprobs:
                    continue
                
                                # Check if this position has both A and B alternatives
                has_A = False
                has_B = False
                temp_prob_A = -float("inf")
                temp_prob_B = -float("inf")
                
                for alt in token_info.top_logprobs:
                    token = alt.token
                    if token == " A":  # Only space + A
                        has_A = True
                        temp_prob_A = alt.logprob
                    elif token == " B":  # Only space + B
                        has_B = True
                        temp_prob_B = alt.logprob
                
                # If we found both A and B at this position, use them
                if has_A and has_B:
                    prob_A = temp_prob_A
                    prob_B = temp_prob_B
                    found_decision = True
                    logger.debug(f"Found decision tokens at position {token_idx}: A={prob_A:.4f}, B={prob_B:.4f}")
                    break
            
            # If we found decision probabilities, normalize and return them
            if found_decision and prob_A != -float("inf") and prob_B != -float("inf"):
                probs = np.exp([prob_A, prob_B])
                probs = probs / np.sum(probs)  # Normalize
                logger.debug(f"OpenAI probabilities - A: {probs[0]:.4f}, B: {probs[1]:.4f}")
                return probs[0], probs[1]
            
            # If no decision found, look for any A or B tokens individually
            logger.debug("No decision position found with both A and B, searching for individual tokens")
            for token_idx in range(len(logprobs.content) - 1, -1, -1):
                token_info = logprobs.content[token_idx]
                if not token_info.top_logprobs:
                    continue
                
                for alt in token_info.top_logprobs:
                    token = alt.token
                    if token == " A" and prob_A == -float("inf"):
                        prob_A = alt.logprob
                    elif token == " B" and prob_B == -float("inf"):
                        prob_B = alt.logprob
                
                # Stop if we found both
                if prob_A != -float("inf") and prob_B != -float("inf"):
                    break
            
            # If we found at least one, try to use them
            if prob_A != -float("inf") or prob_B != -float("inf"):
                if prob_A == -float("inf"):
                    prob_A = prob_B - 2.0  # Assign lower probability
                elif prob_B == -float("inf"):
                    prob_B = prob_A - 2.0  # Assign lower probability
                
                probs = np.exp([prob_A, prob_B])
                probs = probs / np.sum(probs)  # Normalize
                logger.debug(f"OpenAI probabilities (partial) - A: {probs[0]:.4f}, B: {probs[1]:.4f}")
                return probs[0], probs[1]
            
            # Fallback if no choice probabilities found
            logger.debug("No choice probabilities found in OpenAI logprobs, using default 0.5")
            return 0.5, 0.5
            
        except Exception as e:
            logger.error(f"Error extracting choice probabilities from logprobs: {e}")
            return 0.5, 0.5
    
    def _get_batch_choice_probabilities(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        responses: List[Tuple[str, Optional[List[Dict[str, str]]], object]],
        modes: List[str] = None
    ) -> List[Tuple[float, float]]:
        """
        Get choice probabilities for a batch of prompts and responses.
        
        Args:
            prompts: List of input prompts
            responses: List of response tuples from _generate_batch
            modes: List of modes for each prompt/response pair
            
        Returns:
            List of (prob_A, prob_B) tuples
        """
        if self.model_type == "openai":
            # For OpenAI, extract probabilities directly from the logprobs we already have
            batch_probabilities = []
            
            with tqdm(total=len(responses), desc="Extracting probabilities", leave=False) as pbar:
                for response_data in responses:
                    try:
                        # response_data is (response_text, processed_prompt, api_response)
                        if len(response_data) >= 3 and response_data[2] is not None:
                            api_response = response_data[2]
                            prob_A, prob_B = self._extract_choice_probabilities_from_logprobs(api_response)
                        else:
                            # Fallback if no API response available
                            prob_A, prob_B = 0.5, 0.5
                        
                        batch_probabilities.append((prob_A, prob_B))
                    except Exception as e:
                        logger.error(f"Error extracting probabilities: {e}")
                        batch_probabilities.append((0.5, 0.5))
                    
                    pbar.update(1)
            
            return batch_probabilities
        else:
            # For HuggingFace models, use batch forward pass for efficiency
            batch_probabilities = []
            
            if VLLM_AVAILABLE and self.vllm_model is not None:
                # For vLLM, use improved individual probability extraction for each prompt/response
                with tqdm(total=len(prompts), desc="Computing vLLM probabilities", leave=False) as pbar:
                    for i, (prompt, response_data) in enumerate(zip(prompts, responses)):
                        try:
                            response = response_data[0] if isinstance(response_data, tuple) else response_data
                            vllm_output = response_data[2] if isinstance(response_data, tuple) and len(response_data) > 2 else None
                            
                            # Get mode for this specific prompt
                            current_mode = "base"  # Default
                            if modes and i < len(modes):
                                current_mode = modes[i]
                            elif hasattr(self, '_current_mode'):
                                current_mode = self._current_mode
                            
                            # Use the improved probability extraction method
                            prob_A, prob_B = self._get_choice_probabilities_vllm(prompt, response, vllm_output, current_mode)
                            batch_probabilities.append((prob_A, prob_B))
                            
                        except Exception as e:
                            logger.error(f"Error computing probabilities for prompt: {e}")
                            batch_probabilities.append((0.5, 0.5))
                        
                        pbar.update(1)
            else:
                # Check if we have a HuggingFace model available
                if self.model is None:
                    # No HuggingFace model available - return uniform probabilities for all
                    logger.warning("No HuggingFace model available for batch probability calculation - returning uniform probabilities")
                    batch_probabilities = [(0.5, 0.5) for _ in prompts]
                else:
                    # Use batch processing with standard HuggingFace model
                    batch_texts = []
                    valid_indices = []
                    
                    for i, (prompt, response_data) in enumerate(zip(prompts, responses)):
                        try:
                            response = response_data[0] if isinstance(response_data, tuple) else response_data
                            
                            # Ensure prompt is a string for HuggingFace models
                            if not isinstance(prompt, str):
                                logger.error(f"Expected string prompt for HuggingFace model batch processing, got {type(prompt)}")
                                batch_probabilities.append((0.5, 0.5))
                                continue
                            
                            # Find the decision point in the response
                            last_decision_start = response.rfind("I will choose Option")
                            if last_decision_start != -1:
                                full_text = prompt + response[:last_decision_start + len("I will choose Option ")]
                            else:
                                full_text = prompt + " I will choose Option "
                            
                            batch_texts.append(full_text)
                            valid_indices.append(i)
                        except Exception as e:
                            logger.error(f"Error preparing text for batch probability computation {i}: {e}")
                            batch_probabilities.append((0.5, 0.5))
                    
                    if batch_texts:
                        try:
                            # Process in device batches for memory efficiency
                            all_probs = []
                            num_batches = (len(batch_texts) + self.total_batch_size - 1) // self.total_batch_size
                            
                            with tqdm(total=num_batches, desc="Computing probabilities", leave=False) as pbar:
                                for i in range(0, len(batch_texts), self.total_batch_size):
                                    device_batch_texts = batch_texts[i:i + self.total_batch_size]
                                    
                                    # Tokenize device batch
                                    inputs = self.tokenizer(
                                        device_batch_texts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=4096
                                    )
                                    if torch.cuda.is_available():
                                        inputs = {k: v.cuda() for k, v in inputs.items()}
                                    
                                    # Get batch logits
                                    with torch.no_grad():
                                        outputs = self.model(**inputs)
                                        device_batch_logits = outputs.logits[:, -1, :]  # Get logits at last position for each sequence
                                        
                                        # Always probe " A" and " B" tokens (with space prefix) for batch processing
                                        token_A = None
                                        token_B = None
                                        
                                        try:
                                            # Encode " A" and " B" tokens with space prefix
                                            encoded_A = self.tokenizer.encode(" A", add_special_tokens=False)
                                            encoded_B = self.tokenizer.encode(" B", add_special_tokens=False)
                                            
                                            if len(encoded_A) > 0 and len(encoded_B) > 0:
                                                token_A = encoded_A[0]
                                                token_B = encoded_B[0]
                                                logger.debug(f"Batch processing using space-prefixed tokens: ' A' -> {token_A}, ' B' -> {token_B}")
                                            else:
                                                logger.error("Empty token encoding for ' A' and ' B' in batch processing")
                                                # Add default probabilities for this batch
                                                for _ in device_batch_texts:
                                                    all_probs.append([0.5, 0.5])
                                                continue
                                                
                                        except (IndexError, KeyError, Exception) as e:
                                            logger.error(f"Failed to encode ' A' and ' B' tokens in batch processing: {e}")
                                            # Add default probabilities for this batch
                                            for _ in device_batch_texts:
                                                all_probs.append([0.5, 0.5])
                                            continue
                                        
                                        if token_A is not None and token_B is not None:
                                            # Extract logits for A and B tokens
                                            logits_A = device_batch_logits[:, token_A]
                                            logits_B = device_batch_logits[:, token_B]
                                            
                                            # Convert to probabilities
                                            device_batch_probs = torch.softmax(torch.stack([logits_A, logits_B], dim=1), dim=1)
                                            all_probs.extend(device_batch_probs.tolist())
                                        else:
                                            # Add default probabilities for this batch
                                            for _ in device_batch_texts:
                                                all_probs.append([0.5, 0.5])
                                    pbar.update(1)
                            
                            # Insert results at correct positions
                            result_idx = 0
                            for i in range(len(prompts)):
                                if i in valid_indices:
                                    prob_A, prob_B = all_probs[result_idx]
                                    batch_probabilities.append((prob_A, prob_B))
                                    result_idx += 1
                                else:
                                    # This was already added as (0.5, 0.5) above
                                    pass
                        
                        except Exception as e:
                            logger.error(f"Error in batch probability computation: {e}")
                            # Fill remaining with default probabilities
                            while len(batch_probabilities) < len(prompts):
                                batch_probabilities.append((0.5, 0.5))
                
                # Ensure we have the right number of results
                while len(batch_probabilities) < len(prompts):
                    batch_probabilities.append((0.5, 0.5))
        
            return batch_probabilities
    
    def _save_checkpoint(
        self,
        results: List[Dict],
        checkpoint_file: str,
        data_size: str,
        participant_ids: List[str],
        current_mode: str,
        current_progress: int,
        num_examples: int
    ):
        """
        Save checkpoint with current results.
        
        Args:
            results: Current results to save
            checkpoint_file: Path to checkpoint file
            data_size: Size of dataset
            participant_ids: List of participant IDs being processed
            current_mode: Current mode being processed
            current_progress: Current progress (number of trials processed in current mode)
            num_examples: Number of examples used for in-context learning
        """
        checkpoint_metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "data_size": data_size,
            "checkpoint_timestamp": datetime.now().isoformat(),
            "total_results_so_far": len(results),
            "participants_processed": participant_ids,
            "total_participants": len(participant_ids),
            "current_mode": current_mode,
            "current_progress": current_progress,
            "processing_method": "optimized_batch_by_mode_with_checkpoints",
            "is_checkpoint": True,
            "num_examples": num_examples
        }
        
        # Prepare checkpoint data
        checkpoint_data = {
            "metadata": self._make_json_serializable(checkpoint_metadata),
            "results": self._make_json_serializable(results)
        }
        
        # Save checkpoint
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, checkpoint_file: str) -> Tuple[List[Dict], Dict]:
        """
        Load results from a checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Tuple of (results, metadata)
        """
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            results = checkpoint_data.get("results", [])
            metadata = checkpoint_data.get("metadata", {})
            
            logger.info(f"Loaded checkpoint with {len(results)} results from {checkpoint_file}")
            logger.info(f"Checkpoint metadata: {metadata}")
            
            return results, metadata
            
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_file}: {e}")
            return [], {}
    
    def find_latest_checkpoint(self, output_dir: str, data_size: str) -> Optional[str]:
        """
        Find the latest checkpoint file for this model and data size.
        
        Args:
            output_dir: Directory containing checkpoints
            data_size: Size of dataset
            
        Returns:
            Path to latest checkpoint file, or None if no checkpoints found
        """
        model_name_safe = self.model_name.replace("/", "_").replace(":", "_")
        checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{model_name_safe}_{data_size}")
        
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Find all checkpoint files
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith("_results.json"):
                try:
                    # Extract number of results from filename
                    num_results = int(filename.split("_")[1])
                    checkpoint_files.append((num_results, os.path.join(checkpoint_dir, filename)))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoint_files:
            return None
        
        # Return the checkpoint with the most results
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoint_files[0][1]
        
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
     
    def _make_json_serializable(self, obj):
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import pandas as pd
        
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, pd.Series):
            return [self._make_json_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, str):
            return str(obj)
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # NumPy array or pandas object
            return self._make_json_serializable(obj.tolist())
        else:
            return str(obj)
    
    def save_results(
        self,
        results: List[Dict],
        output_file: str,
        data_size: str,
        metadata: Optional[Dict] = None,
        num_examples: int = 0
    ):
        """
        Save results to JSON file with metadata.
        
        Args:
            results: List of result dictionaries
            output_file: Path to output JSON file
            data_size: Size of dataset used (small, large, all)
            metadata: Additional metadata to include
            num_examples: Number of examples used for in-context learning
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare metadata
        final_metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "data_size": data_size,
            "completion_timestamp": datetime.now().isoformat(),
            "total_trials": len(results),
            "modes_tested": ["base", "cot", "human"],
            "num_examples": num_examples
        }
        
        if metadata:
            final_metadata.update(metadata)
        
        # Prepare final output and ensure JSON serializable
        output_data = {
            "metadata": self._make_json_serializable(final_metadata),
            "results": self._make_json_serializable(results)
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
    
    def predict_batch(
        self,
        participants_data: Dict,
        participant_ids: List[str],
        data_size: str,
        output_dir: str = "results",
        save_results: bool = True,
        checkpoint_interval: int = 1000,
        num_examples: int = 1
    ) -> List[Dict]:
        """
        Run predictions for multiple participants and save results.
        Uses optimized batching with checkpoints: processes all trials for each mode together
        and saves checkpoints incrementally for robustness.
        
        Args:
            participants_data: Dictionary containing all participant data
            participant_ids: List of participant IDs to process
            data_size: Size of dataset (small, large, all)
            output_dir: Directory to save results
            save_results: Whether to save results to file
            checkpoint_interval: Number of results to accumulate before saving checkpoint (default: 1000)
            num_examples: Number of examples for in-context learning (default: 1)
            
        Returns:
            List of all results
        """
        # Create checkpoint directory
        if save_results:
            model_name_safe = self.model_name.replace("/", "_").replace(":", "_")
            checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{model_name_safe}_{data_size}_{num_examples}")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine which experiment we're running based on the mode
        # and set appropriate modes to process
        if self.mode.startswith("exp2_"):
            # This is experiment 2 (in-context learning)
            exp2_mode = self.mode[5:]  # Remove "exp2_" prefix
            if exp2_mode == "all":
                modes_to_process = ["base_within_individual", "cot_within_individual", "base_within_context", "cot_within_context"]
            elif exp2_mode == "within_individual":
                modes_to_process = ["base_within_individual", "cot_within_individual"]
            elif exp2_mode == "within_context":
                modes_to_process = ["base_within_context", "cot_within_context"]
            else:
                # Single specific mode
                modes_to_process = [exp2_mode]
        elif self.mode == "all":
            # This is experiment 1 (zero-shot learning)
            modes_to_process = ["base", "cot", "human"]
        else:
            # Single mode specified
            if self.mode in ["within_individual", "within_context"]:
                # This is exp2 calling with simplified mode - expand to full modes
                if self.mode == "within_individual":
                    modes_to_process = ["base_within_individual", "cot_within_individual"]
                else:  # within_context
                    modes_to_process = ["base_within_context", "cot_within_context"]
            elif self.mode in ["base_within_individual", "cot_within_individual", "base_within_context", "cot_within_context"]:
                # This is exp2 calling with full mode names
                modes_to_process = [self.mode]
            else:
                # This is exp1
                modes_to_process = [self.mode]
        
        # Sort participant IDs to ensure consistent order
        participant_ids = sorted(participant_ids)

        # Pre-organize data for faster processing
        print(f"\nPre-organizing data for {data_size} dataset...")

        # Create a mapping of problem_id to participant indices for quick lookup
        problem_to_participants = {}
        for participant_id in participant_ids:
            participant_data = participants_data[participant_id]
            for trial_index, problem_id in enumerate(participant_data["problem_id"]):
                if problem_id not in problem_to_participants:
                    problem_to_participants[problem_id] = []
                problem_to_participants[problem_id].append((participant_id, trial_index))

        # Get sorted unique problem IDs
        unique_problem_ids = sorted(problem_to_participants.keys())
        num_trials = len(unique_problem_ids)

        # Prepare all data for this dataset, maintaining hierarchical order
        all_prompts = []
        all_metadata = []

        print(f"Organizing data in order: mode -> problem_id -> participant")
        print(f"Mode order: {', '.join(modes_to_process)}")
        print(f"Number of unique trials across all participants: {num_trials}")
        print(f"Number of participants: {len(participant_ids)}")

        # First organize by mode
        for mode in modes_to_process:
            print(f"\nPreparing {mode} mode data...")
            # Then by problem ID
            for problem_id in unique_problem_ids:
                print(f"  Processing problem ID {problem_id}")
                # Get all participants who have this problem_id
                for participant_id, trial_index in problem_to_participants[problem_id]:
                    participant_data = participants_data[participant_id]
                    try:
                        # For exp2 modes, we need to prepare examples
                        examples = None
                        example_trial_ids = []
                        example_participant_ids = []
                        
                        if "within_individual" in mode:
                            # Prepare within-individual examples
                            examples = self._prepare_within_individual_examples(
                                participant_data, trial_index, participants_data, participant_id, num_examples
                            )
                            example_trial_ids = [str(ex.get("trial_id", "")) for ex in examples]
                            example_participant_ids = [str(participant_id) for _ in examples]
                        elif "within_context" in mode:
                            # Prepare within-context examples
                            examples = self._prepare_within_context_examples(
                                participants_data, participant_id, problem_id, trial_index, num_examples
                            )
                            example_trial_ids = [str(problem_id) for _ in examples]
                            example_participant_ids = [str(ex.get("participant_id", "")) for ex in examples]
                        
                        # Format examples into string
                        formatted_examples = self._format_examples_for_prompt(examples) if examples else None
                        
                        # Format prompt for this trial and mode
                        prompt = self._format_prompt(
                            mode,
                            participant_data["probabilities"][trial_index],
                            participant_data["outcomes"][trial_index],
                            participant_data["think_aloud"][trial_index],
                            participant_data["question_context"][trial_index],
                            formatted_examples,
                            "think_aloud"  # Default example_type for predict_batch
                        )
                        
                        all_prompts.append(prompt)
                        all_metadata.append({
                            "participant_id": participant_id,
                            "trial_id": problem_id,  # Use the actual problem_id
                            "mode": mode,
                            "participant_data": participant_data,
                            "trial_index": trial_index,  # Store the index for later use
                            "example_trial_ids": example_trial_ids,
                            "example_participant_ids": example_participant_ids
                        })
                        
                    except Exception as e:
                        logger.error(f"Error formatting prompt for participant {participant_id}, problem ID {problem_id}, mode {mode}: {e}")
                        # Add placeholder for error
                        all_prompts.append("")
                        all_metadata.append({
                            "participant_id": participant_id,
                            "trial_id": problem_id,  # Use the actual problem_id
                            "mode": mode,
                            "participant_data": participant_data,
                            "trial_index": trial_index,  # Store the index for later use
                            "error": str(e)
                        })
        
        # Split into batches while maintaining the prepared order
        total_trials = len(all_prompts)
        num_batches = (total_trials + checkpoint_interval - 1) // checkpoint_interval
        
        print(f"\nProcessing {data_size} dataset in {num_batches} batches of {checkpoint_interval} trials each...")
        
        all_results = []
        total_results_processed = 0
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * checkpoint_interval
            end_idx = min((batch_idx + 1) * checkpoint_interval, total_trials)
            
            batch_prompts = all_prompts[start_idx:end_idx]
            batch_metadata = all_metadata[start_idx:end_idx]
            
            # Log the content of this batch
            batch_modes = set(meta["mode"] for meta in batch_metadata)
            batch_trials = set(meta["trial_id"] for meta in batch_metadata)
            batch_participants = set(meta["participant_id"] for meta in batch_metadata)
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} trials)...")
            print(f"Modes in this batch: {', '.join(sorted(batch_modes))}")
            print(f"Trial IDs in this batch: {', '.join(map(str, sorted(batch_trials)))}")
            print(f"Participants in this batch: {', '.join(map(str, sorted(batch_participants)))}")
            
            try:
                # Generate responses for all trials in this batch
                batch_responses = self._generate_batch(batch_prompts, batch_metadata[0]["mode"])  # Use the mode from the first item in batch
                
                # Get choice probabilities for all trials in this batch
                # Extract modes for this batch
                batch_modes = [meta["mode"] for meta in batch_metadata]
                batch_probabilities = self._get_batch_choice_probabilities(batch_prompts, batch_responses, batch_modes)
                
                # Process results for all trials in this batch
                batch_results = []
                with tqdm(total=len(batch_responses), desc=f"Processing batch {batch_idx + 1}", leave=False) as pbar:
                    for i, (response_data, metadata, (prob_A, prob_B)) in enumerate(zip(batch_responses, batch_metadata, batch_probabilities)):
                        try:
                            if "error" in metadata:
                                # Handle error case
                                trial_result = {
                                    "sub_id": metadata["participant_id"],
                                    "trial_id": metadata["trial_id"],
                                    "mode": metadata["mode"],
                                    "think_aloud": metadata["participant_data"]["think_aloud"][metadata["trial_index"]],
                                    "actual_choice": metadata["participant_data"]["choices"][metadata["trial_index"]],
                                    "extracted_choice": None,
                                    "probability_option_a": 0.5,
                                    "probability_option_b": 0.5,
                                    "cot": None,
                                    "full_generation": f"ERROR: {metadata['error']}",
                                    "question_context": metadata["participant_data"]["question_context"][metadata["trial_index"]],
                                    "example_trial_ids": metadata.get("example_trial_ids", []),
                                    "example_participant_ids": metadata.get("example_participant_ids", []),
                                    "pa": metadata["participant_data"]["probabilities"][metadata["trial_index"]][0],
                                    "va": metadata["participant_data"]["outcomes"][metadata["trial_index"]][0],
                                    "pb": metadata["participant_data"]["probabilities"][metadata["trial_index"]][1],
                                    "vb": metadata["participant_data"]["outcomes"][metadata["trial_index"]][1]
                                }
                            else:
                                # Process successful response
                                response = self._process_response(response_data, metadata["mode"])
                                choice = self._extract_choice(response)
                                
                                # Determine CoT content based on mode
                                if metadata["mode"] == "base":
                                    cot_content = None
                                    full_generation = response
                                elif metadata["mode"] == "cot":
                                    cot_content = response
                                    full_generation = response
                                else:  # human mode
                                    cot_content = metadata["participant_data"]["think_aloud"][metadata["trial_index"]]
                                    full_generation = response  # Use the model's response which should be just the choice
                                
                                trial_result = {
                                    "sub_id": metadata["participant_id"],
                                    "trial_id": metadata["trial_id"],
                                    "mode": metadata["mode"],
                                    "think_aloud": metadata["participant_data"]["think_aloud"][metadata["trial_index"]],
                                    "actual_choice": metadata["participant_data"]["choices"][metadata["trial_index"]],
                                    "extracted_choice": choice,
                                    "probability_option_a": prob_A,
                                    "probability_option_b": prob_B,
                                    "cot": cot_content,
                                    "full_generation": full_generation,
                                    "question_context": metadata["participant_data"]["question_context"][metadata["trial_index"]],
                                    "example_trial_ids": metadata.get("example_trial_ids", []),
                                    "example_participant_ids": metadata.get("example_participant_ids", []),
                                    "pa": metadata["participant_data"]["probabilities"][metadata["trial_index"]][0],
                                    "va": metadata["participant_data"]["outcomes"][metadata["trial_index"]][0],
                                    "pb": metadata["participant_data"]["probabilities"][metadata["trial_index"]][1],
                                    "vb": metadata["participant_data"]["outcomes"][metadata["trial_index"]][1]
                                }
                            
                            batch_results.append(trial_result)
                            
                        except Exception as e:
                            logger.error(f"Error processing result {i} in batch {batch_idx + 1}: {e}")
                            # Add error result
                            error_result = {
                                "sub_id": metadata["participant_id"],
                                "trial_id": metadata["trial_id"],
                                "mode": metadata["mode"],
                                "think_aloud": metadata["participant_data"]["think_aloud"][metadata["trial_index"]],
                                "actual_choice": metadata["participant_data"]["choices"][metadata["trial_index"]],
                                "extracted_choice": None,
                                "probability_option_a": 0.5,
                                "probability_option_b": 0.5,
                                "cot": None,
                                "full_generation": f"ERROR: {str(e)}",
                                "question_context": metadata["participant_data"]["question_context"][metadata["trial_index"]],
                                "example_trial_ids": metadata.get("example_trial_ids", []),
                                "example_participant_ids": metadata.get("example_participant_ids", []),
                                "pa": metadata["participant_data"]["probabilities"][metadata["trial_index"]][0],
                                "va": metadata["participant_data"]["outcomes"][metadata["trial_index"]][0],
                                "pb": metadata["participant_data"]["probabilities"][metadata["trial_index"]][1],
                                "vb": metadata["participant_data"]["outcomes"][metadata["trial_index"]][1]
                            }
                            batch_results.append(error_result)
                        
                        pbar.update(1)
                
                # Add batch results to all results
                all_results.extend(batch_results)
                total_results_processed += len(batch_results)
                
                # Save checkpoint after each batch
                if save_results:
                    checkpoint_file = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_{total_results_processed}_results.json"
                    )
                    self._save_checkpoint(
                        all_results,
                        checkpoint_file,
                        data_size,
                        participant_ids,
                        f"batch_{batch_idx + 1}",
                        total_results_processed,
                        num_examples
                    )
                    print(f"Saved checkpoint with {total_results_processed} total results")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                # Add error results for all trials in this batch
                for metadata in batch_metadata:
                    error_result = {
                        "sub_id": metadata["participant_id"],
                        "trial_id": metadata["trial_id"],
                        "mode": metadata["mode"],
                        "think_aloud": metadata["participant_data"]["think_aloud"][metadata["trial_index"]],
                        "actual_choice": metadata["participant_data"]["choices"][metadata["trial_index"]],
                        "extracted_choice": None,
                        "probability_option_a": 0.5,
                        "probability_option_b": 0.5,
                        "cot": None,
                        "full_generation": f"BATCH_ERROR: {str(e)}",
                        "question_context": metadata["participant_data"]["question_context"][metadata["trial_index"]],
                        "example_trial_ids": metadata.get("example_trial_ids", []),
                        "example_participant_ids": metadata.get("example_participant_ids", []),
                        "pa": metadata["participant_data"]["probabilities"][metadata["trial_index"]][0],
                        "va": metadata["participant_data"]["outcomes"][metadata["trial_index"]][0],
                        "pb": metadata["participant_data"]["probabilities"][metadata["trial_index"]][1],
                        "vb": metadata["participant_data"]["outcomes"][metadata["trial_index"]][1]
                    }
                    all_results.append(error_result)
                    total_results_processed += 1
                
                # Save checkpoint even after batch error
                if save_results:
                    checkpoint_file = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_{total_results_processed}_results.json"
                    )
                    self._save_checkpoint(
                        all_results,
                        checkpoint_file,
                        data_size,
                        participant_ids,
                        f"batch_{batch_idx + 1}_error",
                        total_results_processed,
                        num_examples
                    )
                    print(f"Saved checkpoint with {total_results_processed} total results after batch error")
        
        # Save final results if requested
        if save_results:
            # Generate filename
            output_file = os.path.join(
                output_dir,
                f"{model_name_safe}_{data_size}_{num_examples}_results.json"
            )
            
            # Save final results
            self.save_results(
                all_results,
                output_file,
                data_size,
                metadata={
                    "participants_processed": participant_ids,
                    "total_participants": len(participant_ids),
                    "processing_method": "optimized_batch_with_checkpoints",
                    "total_batch_size": self.total_batch_size,
                    "num_workers": self.num_workers,
                    "checkpoint_interval": checkpoint_interval,
                    "total_results_processed": total_results_processed
                },
                num_examples=num_examples
            )
            
            print(f"\nResults saved to {output_file}")
        
        return all_results

    def _prepare_within_individual_examples(
        self,
        participant_data: Dict,
        test_trial_index: int,
        participants_data: Dict,
        participant_id: str,
        num_examples: int,
        random_state: int = 2024
    ) -> List[Dict]:
        """
        Prepare examples for within-individual learning.
        
        Args:
            participant_data: Dictionary containing participant's data
            test_trial_index: Index of the test trial
            participants_data: All participants data (not used here but kept for consistency)
            participant_id: ID of the participant
            num_examples: Number of examples to prepare
            random_state: Random seed for reproducibility
            
        Returns:
            List of example dictionaries
        """
        import random
        
        # Get all trial indices except the test trial
        train_indices = [i for i in range(len(participant_data["problem_id"])) if i != test_trial_index]
        
        # Randomly select examples
        random.seed(random_state)
        selected_indices = random.sample(train_indices, min(num_examples, len(train_indices)))
        
        # Prepare examples
        examples = []
        for idx in selected_indices:
            example = {
                "trial_id": participant_data["problem_id"][idx],
                "probabilities": participant_data["probabilities"][idx],
                "outcomes": participant_data["outcomes"][idx],
                "think_aloud": participant_data["think_aloud"][idx],
                "question_context": participant_data["question_context"][idx],
                "choice": participant_data["choices"][idx],  # Add choice to example
                "participant_id": participant_id
            }
            examples.append(example)
        
        return examples
    
    def _prepare_within_context_examples(
        self,
        all_participants_data: Dict,
        test_participant_id: str,
        test_trial_id: str,
        test_trial_index: int,
        num_examples: int,
        random_state: int = 2024
    ) -> List[Dict]:
        """
        Prepare examples for within-context learning by finding trials with the same ID
        from other participants.
        
        Args:
            all_participants_data: Dictionary containing all participants' data
            test_participant_id: ID of the test participant
            test_trial_id: ID of the test trial
            test_trial_index: Index of the test trial (not used but kept for consistency)
            num_examples: Number of examples to prepare
            random_state: Random seed for reproducibility
            
        Returns:
            List of example dictionaries
        """
        import random
        import numpy as np
        
        # Find participants who have the same trial ID
        available_examples = []
        for participant_id, participant_data in all_participants_data.items():
            if participant_id == test_participant_id:
                continue
                
            # Find the index of the trial in this participant's data
            trial_ids = participant_data["problem_id"]
            if test_trial_id in trial_ids:
                trial_index = np.where(trial_ids == test_trial_id)[0][0]
                available_examples.append({
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "participant_data": participant_data
                })
        
        if len(available_examples) == 0:
            return []
        
        # Randomly select examples
        random.seed(random_state)
        selected_examples = random.sample(available_examples, min(num_examples, len(available_examples)))
        
        # Format examples
        examples = []
        for example in selected_examples:
            participant_id = example["participant_id"]
            trial_index = example["trial_index"]
            participant_data = example["participant_data"]
            
            example_dict = {
                "trial_id": test_trial_id,
                "probabilities": participant_data["probabilities"][trial_index],
                "outcomes": participant_data["outcomes"][trial_index],
                "think_aloud": participant_data["think_aloud"][trial_index],
                "question_context": participant_data["question_context"][trial_index],
                "choice": participant_data["choices"][trial_index],  # Add choice to example
                "participant_id": participant_id
            }
            examples.append(example_dict)
        
        return examples
    
    def _format_examples_for_prompt(self, examples: List[Dict], example_type: str = "think_aloud") -> str:
        """
        Format examples into a string for the prompt.
        
        Args:
            examples: List of example dictionaries
            example_type: Type of information to include ("think_aloud", "choice", "both")
            
        Returns:
            Formatted examples string
        """
        if not examples:
            return ""
        
        # Validate example_type
        if example_type not in ["think_aloud", "choice", "both"]:
            raise ValueError(f"example_type must be one of ['think_aloud', 'choice', 'both'], got {example_type}")
        
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"Example {i}:\n"
            formatted_example += f"Decision Scenario:\n{example['question_context']}\n"
            # Always present think-aloud first, then choice when both are configured
            if example_type in ["think_aloud", "both"] and example.get('think_aloud'):
                formatted_example += f"Think-aloud:\n{example['think_aloud']}\n"
            if example_type in ["choice", "both"] and 'choice' in example:
                choice_text = "A" if example['choice'] == 0 else "B"
                formatted_example += f"I chose Option {choice_text}.\n"
            formatted_examples.append(formatted_example)
        return "\n".join(formatted_examples)
        return "\n".join(formatted_examples)