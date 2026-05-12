"""Colab launcher: opens an ngrok tunnel, then runs uvicorn in the foreground.

Usage in a Colab cell:

    !pip install -r /content/GUI-Actor-V2/demo/requirements.txt pyngrok
    %env NGROK_AUTHTOKEN=your_token_here
    %cd /content/GUI-Actor-V2/demo
    !python launch_colab.py

The public URL is printed before uvicorn starts; copy it and hit /health.
"""

import os
import sys

from pyngrok import ngrok
import uvicorn

PORT = int(os.environ.get("PORT", "7860"))


def main() -> int:
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print(
            "ERROR: set NGROK_AUTHTOKEN before running. Get one at "
            "https://dashboard.ngrok.com/get-started/your-authtoken",
            file=sys.stderr,
        )
        return 1

    ngrok.set_auth_token(token)
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)

    tunnel = ngrok.connect(PORT, "http")
    print("=" * 60)
    print(f"Public URL: {tunnel.public_url}")
    print(f"Health:     {tunnel.public_url}/health")
    print(f"Predict:    POST {tunnel.public_url}/predict")
    print("=" * 60, flush=True)

    uvicorn.run("server:app", host="0.0.0.0", port=PORT, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
