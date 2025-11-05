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

    # Extract only the newly generated tokens (not the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]

    # Decode only the newly generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # If the model generated text that looks like it's repeating the prompt structure,
    # filter it out. Look for patterns like "system\n", "user\n", "assistant\n" at the start
    lines = generated_text.split("\n")
    cleaned_lines = []
    in_assistant_response = False

    for line in lines:
        # Remove any special token markers from the line
        line = line.replace("<|im_start|>", "").replace("<|im_end|>", "")
        line_stripped = line.strip()
        if not line_stripped:
            if in_assistant_response:
                cleaned_lines.append("")
            continue

        # Skip role markers
        if line_stripped.lower() in ("system", "user", "assistant"):
            if line_stripped.lower() == "assistant":
                in_assistant_response = True
            continue

        # Skip lines that start with role markers followed by colon or content
        if ":" in line_stripped:
            role_prefix = line_stripped.split(":")[0].strip().lower()
            if role_prefix in ("system", "user", "assistant"):
                # If it's assistant, take the content after the colon
                if role_prefix == "assistant":
                    content = ":".join(line_stripped.split(":")[1:]).strip()
                    if content:
                        cleaned_lines.append(content)
                        in_assistant_response = True
                continue

        # If we've seen assistant marker or are in response, include the line
        if in_assistant_response or not any(
            keyword in line_stripped.lower()
            for keyword in [
                "you are",
                "guidelines",
                "remember:",
                "query:",
                "context documents",
            ]
        ):
            cleaned_lines.append(line)
            in_assistant_response = True

    if cleaned_lines:
        generated_text = "\n".join(cleaned_lines).strip()

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

        # Extract only the newly generated tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]

        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Clean up: remove any end markers and trailing whitespace
        generated_text = generated_text.replace("<|im_end|>", "").strip()

        # Simulate streaming by yielding in small chunks
        chunk_size = 5  # characters per chunk
        for i in range(0, len(generated_text), chunk_size):
            yield generated_text[i : i + chunk_size]
