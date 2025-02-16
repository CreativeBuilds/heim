import getpass

from helm.common.authentication import Authentication
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.request import Request, RequestResult, TextToImageRequest
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.accounts import Account
from helm.proxy.services.remote_service import RemoteService

# An example of how to use the request API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
account: Account = service.get_account(auth)
print(account.usages)

# Make a request
request = Request(model="ai21/j1-large", prompt="Life is like a box of", echo_prompt=True)
request_result: RequestResult = service.make_request(auth, request)
print(request_result.completions[0].text)

# Expect different responses for the same request but with different values for `random`.
# Passing in the same value for `random` guarantees the same results.
request = Request(prompt="Life is like a box of", random="1")
request_result = service.make_request(auth, request)
print(request_result.completions[0].text)

# How to get the embedding for some text
request = Request(model="openai/text-similarity-ada-001", prompt="Life is like a box of", embedding=True)
request_result = service.make_request(auth, request)
print(request_result.embedding)

# Tokenize
request = TokenizationRequest(tokenizer="ai21/j1-jumbo", text="Tokenize me please.")
tokenization_request_result: TokenizationRequestResult = service.tokenize(auth, request)
print(f"Number of tokens: {len(tokenization_request_result.tokens)}")

# Calculate toxicity scores
text = "you suck."
request = PerspectiveAPIRequest(text_batch=[text])
perspective_request_result: PerspectiveAPIRequestResult = service.get_toxicity_scores(auth, request)
print(f"{text} - toxicity score: {perspective_request_result.text_to_toxicity_attributes[text].toxicity_score}")

# Image generation
request = TextToImageRequest(
    model="openai/dalle-2", prompt="A sheep flying over the moon", guidance_scale=7.5, num_completions=3
)
request_result = service.make_request(auth, request)
print(f"Image saved at {request_result.completions[0].file_location} for prompt: {request.prompt}")

# Moderation API
request = ModerationAPIRequest(text="A man repeatedly stabs his victim causing him to bleed out.")
request_result: ModerationAPIRequestResult = service.get_moderation_results(auth, request)
print(request_result.flagged_results)
