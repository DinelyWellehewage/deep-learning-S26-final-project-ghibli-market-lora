from diffusers import StableDiffusionPipeline

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"

print("Loading Stable Diffusion 1.5...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    dtype="auto",
)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

print("Original tokenizer size:", len(tokenizer))
print("Original text embedding size:", text_encoder.get_input_embeddings().weight.shape)

num_added_tokens = tokenizer.add_tokens(INSTANCE_TOKEN)

if num_added_tokens == 0:
    raise ValueError(f"Token {INSTANCE_TOKEN} already exists in tokenizer.")

text_encoder.resize_token_embeddings(len(tokenizer))

token_id = tokenizer.convert_tokens_to_ids(INSTANCE_TOKEN)

print("Added token:", INSTANCE_TOKEN)
print("Token ID:", token_id)
print("New tokenizer size:", len(tokenizer))
print("New text embedding size:", text_encoder.get_input_embeddings().weight.shape)

print("Token added successfully.")