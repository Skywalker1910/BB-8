# Training and AWS Deployment

BB8 is trained on a local NVIDIA GPU and deployed to AWS for inference. The
older from-scratch checkpoints are small enough for simple CPU Lambda testing.
The `v004-dev` adapter uses a 0.5B pretrained base model plus LoRA weights, so
it is a more realistic model but also a larger deployment artifact with slower
CPU cold starts.

## Architecture

```text
Local RTX GPU                AWS
---------------              ------------------------------
training corpus              API Gateway HTTP API
      |                               |
run_experiment.py / fine_tune.py       v
      |                      Lambda container (CPU)
checkpoint or adapter + tokenizer      |
      |                               v
package_model.py              packaged BB8 model
```

The model is embedded in the container image. This keeps the educational
deployment easy to reproduce and avoids adding S3 access to the runtime.

## 1. Train and verify a model

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe experiments\run_experiment.py `
  --config configs\small_model.yaml `
  --data data\tiny_shakespeare.txt `
  --name bb8-char-small-v002 `
  --prompt "ROMEO:"
```

Use a new run name for every experiment. The runner will not overwrite an
existing final checkpoint.

For the pretrained adapter track:

```powershell
.\.venv\Scripts\python.exe fine_tune.py `
  --config configs\qwen_lora_v004.yaml `
  --name bb8-qwen-lora-v004-dev
```

## 2. Test generation locally

```powershell
.\.venv\Scripts\python.exe generate.py `
  --model-dir checkpoints\bb8-char-small-v002 `
  --prompt "ROMEO:" `
  --max-new-tokens 150
```

## 3. Stage the selected model

```powershell
.\.venv\Scripts\python.exe scripts\package_model.py `
  --model-name bb8-char-small-v002
```

For a LoRA adapter, use the adapter run name:

```powershell
.\.venv\Scripts\python.exe scripts\package_model.py `
  --model-name bb8-qwen-lora-v004-dev
```

This copies the selected checkpoint or adapter, tokenizer files, metadata, and a
checksum manifest into `deployment/model`. Generated artifacts in that directory
are ignored by Git.

## 4. Exercise the HTTP API locally

```powershell
.\.venv\Scripts\python.exe -m api.local_server `
  --model-dir checkpoints\bb8-qwen-lora-v004-dev `
  --api-key local-test-key `
  --device cuda
```

In another PowerShell window:

```powershell
$headers = @{ "X-API-Key" = "local-test-key" }
$body = @{
  prompt = "ROMEO:"
  max_new_tokens = 100
  strategy = "top_p"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/generate `
  -Headers $headers `
  -ContentType application/json `
  -Body $body
```

Open `http://127.0.0.1:8000/chat` for the local browser interface. The same
conversation contract is available directly through `POST /chat`:

```powershell
$chatBody = @{
  messages = @(
    @{ role = "user"; content = "Hello BB8" }
  )
  max_new_tokens = 80
  strategy = "top_p"
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -Headers @{ "X-API-Key" = "local-test-key" } `
  -ContentType application/json `
  -Body $chatBody
```

## 5. Install and configure AWS tools

Install Docker Desktop, AWS CLI v2, and AWS SAM CLI. Configure a dedicated IAM
deployment identity rather than using root-account credentials:

```powershell
aws login
aws sts get-caller-identity
sam --version
docker --version
```

If your organization uses IAM Identity Center, use `aws configure sso`
instead of `aws login`.

Do not commit AWS keys or the API key to this repository.

## 6. Build and deploy

Generate an API key and let SAM create the ECR repository, Lambda function,
API Gateway routes, log permissions, and CloudFormation stack:

```powershell
$apiKey = [guid]::NewGuid().ToString("N")
sam build
sam deploy --guided `
  --parameter-overrides `
    ApiKey=$apiKey `
    AllowedOrigin=https://www.adityamore.dev
```

If a managed network intercepts HTTPS, export its public trusted root
certificate to the git-ignored `deployment/certs` directory before building.
Do not disable TLS verification, and never place a private certificate there:

```powershell
Copy-Item C:\path\to\trusted-root-ca.crt deployment\certs\custom-root.crt
sam build
```

Suggested guided-deployment values:

- Stack name: `bb8-api`
- Region: the same region used by the portfolio application
- Confirm changes before deploy: `Y`
- Allow SAM to create IAM roles: `Y`
- Save arguments to `samconfig.toml`: `Y`

The deployment output contains `ApiUrl`.

## 7. Test the deployed endpoint

```powershell
$apiUrl = "https://replace-with-api-id.execute-api.region.amazonaws.com"

Invoke-RestMethod `
  -Method Get `
  -Uri "$apiUrl/health"

Invoke-RestMethod `
  -Method Post `
  -Uri "$apiUrl/generate" `
  -Headers @{ "X-API-Key" = $apiKey } `
  -ContentType application/json `
  -Body '{"prompt":"ROMEO:","max_new_tokens":100,"strategy":"top_p"}'
```

Keep `BB8_API_KEY` only in the server-side environment of the portfolio app.
Never expose it through a browser-prefixed public environment variable.

## 8. Remove the cloud resources

When the endpoint is no longer needed:

```powershell
sam delete --stack-name bb8-api
```

Deleting the stack removes the function and API. Review ECR and CloudWatch in
the AWS console afterward in case retained images or logs remain.
