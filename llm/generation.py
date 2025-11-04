"""
Language model inference for RAG response generation.
Handles tokenization, device placement, and streaming output
for conversational responses based on retrieved context.
"""


def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    """Generate response from prompt using model and tokenizer.

    Args:
        tokenizer: Tokenizer instance.
        model: Model instance.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        device: Device string ("cpu", "cuda", etc.).

    Returns:
        Generated text with special tokens stripped.

    Note:
        Moves tensors to specified device, performs deterministic cleanup,
        and strips special tokens from output.
    """
    import torch

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to device
    # Check if device_map="auto" is being used (model handles device automatically)
    has_device_map = hasattr(model, "hf_device_map") or (
        hasattr(model, "device") and str(model.device) != "cpu"
    )

    if has_device_map:
        # Model manages device placement automatically
        # Put inputs on the device of the first parameter
        try:
            first_param = next(iter(model.parameters()))
            input_device = str(first_param.device)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        except StopIteration:
            # Fallback: use provided device
            inputs = {k: v.to(device) for k, v in inputs.items()}
    elif device != "cpu":
        # Single GPU: move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if hasattr(model, "to"):
            model = model.to(device)
    else:
        # CPU: ensure inputs and model are on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        torch.set_num_threads(6)
        if hasattr(model, "to"):
            model = model.to("cpu")

    # Generate response
    with (
        torch.no_grad()
    ):  # Deterministic cleanup: disable gradient computation
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and strip special tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output if present
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].strip()

    return generated_text


def generate_response_streaming(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
):
    """Generate response with streaming support.

    Args:
        tokenizer: Tokenizer instance.
        model: Model instance.
        prompt: Input prompt text.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        device: Device string ("cpu", "cuda", etc.).

    Yields:
        Generated text chunks (incremental).

    Note:
        Moves tensors to specified device, performs deterministic cleanup,
        and strips special tokens from output.
    """
    import threading

    import torch

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to device
    has_device_map = hasattr(model, "hf_device_map") or (
        hasattr(model, "device") and str(model.device) != "cpu"
    )

    if has_device_map:
        try:
            first_param = next(iter(model.parameters()))
            input_device = str(first_param.device)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        except StopIteration:
            inputs = {k: v.to(device) for k, v in inputs.items()}
    elif device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if hasattr(model, "to"):
            model = model.to(device)
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        torch.set_num_threads(6)
        if hasattr(model, "to"):
            model = model.to("cpu")

    # Try to use TextIteratorStreamer if available
    try:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Generate in a separate thread
        generation_thread = threading.Thread(
            target=model.generate, kwargs=generation_kwargs
        )
        generation_thread.start()

        # Yield tokens as they come
        yield from streamer

        generation_thread.join()

    except (ImportError, AttributeError):
        # Fallback: generate all at once and simulate streaming
        # This is less ideal but works if TextIteratorStreamer is not available
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode full output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        # Simulate streaming by yielding in small chunks
        chunk_size = 5  # characters per chunk
        for i in range(0, len(generated_text), chunk_size):
            yield generated_text[i : i + chunk_size]
