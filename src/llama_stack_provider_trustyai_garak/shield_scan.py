"""This contains the implementation of passing garak probes to shields before and after sending prompts to LLM"""

from typing import List, Union
import httpx
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff

from llama_stack.apis.safety import RunShieldResponse, ViolationLevel
from llama_stack.apis.inference import OpenAIChatCompletion
import logging

logger = logging.getLogger(__name__)

OPENAI_COMPATIBLE_INFERENCE_URI = "/openai/v1/chat/completions"
RUN_SHIELD_URI = "/safety/run-shield"

## from llama_stack/providers/inline/safety/llama_guard/llama_guard.py
CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"

# Check if GARAK_LOG_FILE environment variable is set
garak_log_file = os.getenv("GARAK_LOG_FILE")
if garak_log_file and not logger.handlers:
    # Create file handler for the garak log file
    file_handler = logging.FileHandler(garak_log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"Shield scan logger configured to write to: {garak_log_file}")



class SimpleShieldOrchestrator:
    def __init__(self):
        self._client = httpx.Client(
            timeout=30, # TODO: make this configurable
            limits=httpx.Limits(
                max_connections=20, # TODO: make this configurable
                max_keepalive_connections=10, # TODO: make this configurable
            ),
            http2=True
        )

    @backoff.on_exception(
            backoff.fibo, 
            (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError), 
            max_value=50,
        )
    def _get_shield_response(self, shield_id: str, prompt: str, base_url: str, params: dict={}) -> RunShieldResponse:
        shield_run_url = f"{base_url}{RUN_SHIELD_URI}"
        try:
            response = self._client.post(
                shield_run_url,
                json={
                    "shield_id": shield_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "params": params
                    }
                )
            logger.debug(f"Shield response status code: {response.status_code}")
            return RunShieldResponse(**response.json())
        except Exception as e:
            logger.error(f"Error running shield: {e}")
            raise e
    
    def _run_shields_with_early_exit(self, shield_ids: List[str], prompt: str, base_url: str, params: dict={}, max_workers: int=5) -> bool:
        """Run multiple shields in parallel and return True if any shield returns a violation"""
        with ThreadPoolExecutor(max_workers=min(max_workers, len(shield_ids))) as executor:
            futures_to_shields = {executor.submit(self._get_shield_response, shield_id, prompt, base_url, params): shield_id for shield_id in shield_ids}
            for future in as_completed(futures_to_shields):
                shield_id = futures_to_shields[future]
                try:
                    shield_response = future.result()
                    if shield_response.violation and shield_response.violation.violation_level == ViolationLevel.ERROR:
                        # cancel pending futures
                        for future in futures_to_shields:
                            future.cancel()
                        return True
                except Exception as e:
                    logger.error(f"Error running shield {shield_id}: {e}")
                    return False
        return False

    @backoff.on_exception(
            backoff.fibo, 
            (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError), 
            max_value=70,
        )
    def _get_LLM_response(self, model: str, prompt: str, base_url: str) -> OpenAIChatCompletion:
        inference_url = f"{base_url}{OPENAI_COMPATIBLE_INFERENCE_URI}"
        try:
            response = self._client.post(
                inference_url,
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )
            logger.debug(f"LLM response status code: {response.status_code}")
            return OpenAIChatCompletion(**response.json())
        except Exception as e:
            logger.error(f"Error running LLM: {e}")
            raise e

    def __call__(self, prompt: str, **kwargs) -> List[Union[str, None]]:
        """
        Orchestrate the shield scan and return the response
        """
        input_shields: List[str] = kwargs["llm_io_shield_mapping"]["input"]
        output_shields: List[str] = kwargs["llm_io_shield_mapping"]["output"]

        if input_shields:
            logger.debug(f"Running input shields: {input_shields}")
            if self._run_shields_with_early_exit(input_shields, prompt, kwargs["base_url"], kwargs.get("params", {}), max_workers=kwargs.get("max_workers", 5)):
                return [CANNED_RESPONSE_TEXT]
            # for shield_id in input_shields:
            #     shield_response = self._get_shield_response(shield_id, prompt, kwargs["base_url"], kwargs.get("params", {}))
            #     if shield_response.violation and shield_response.violation.violation_level == ViolationLevel.ERROR:
            #         return [CANNED_RESPONSE_TEXT]
            logger.debug(f"No input violation detected. Running LLM")
        else:
            logger.debug(f"No input shields detected. Running LLM")

        model_response = self._get_LLM_response(
            model=kwargs["model"],
            prompt=prompt,
            base_url=kwargs["base_url"]
        )
        llm_response: str = model_response.choices[0].message.content

        if output_shields:
            logger.debug(f"Running output shields: {output_shields}")
            if self._run_shields_with_early_exit(output_shields, llm_response, kwargs["base_url"], kwargs.get("params", {}), max_workers=kwargs.get("max_workers", 5)):
                return [CANNED_RESPONSE_TEXT]
            # for shield_id in output_shields:
            #     shield_response = self._get_shield_response(shield_id, llm_response, kwargs["base_url"], kwargs.get("params", {}))
            #     if shield_response.violation and shield_response.violation.violation_level == ViolationLevel.ERROR:
            #         return [CANNED_RESPONSE_TEXT]
            logger.debug(f"No output violation detected. Returning LLM response")
        else:
            logger.debug(f"No output shields detected. Returning LLM response")
        
        return [llm_response]

    def close(self):
        if hasattr(self, '_client'):
            self._client.close()
    
    def __del__(self):
        self.close()

simple_shield_orchestrator = SimpleShieldOrchestrator()