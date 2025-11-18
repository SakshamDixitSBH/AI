import os
import subprocess

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from google.oauth2.credentials import Credentials


def get_coin_token() -> str:
    """
    Uses Helix to obtain an access token.
    Adjust the command if your Helix CLI differs.
    """
    command = "helix auth access-token print -a"
    result = subprocess.check_output(command, shell=True)
    return result.decode().strip()


# ------------------------------------------------------------------
# Network / cert setup – keep these aligned with your environment
# ------------------------------------------------------------------

# Internal CA chain (adjust path if needed)
os.environ["REQUESTS_CA_BUNDLE"] = (
    f"{os.environ.get('HOME', '')}/Downloads/CitiInternalCAChain_PROD.pem"
)

# Clear proxies for this session (if your environment requires that)
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


class VertexGenAI:
    """
    Wrapper around the Citi R2D2 / Vertex GenAI endpoint.

    Responsibility:
      - Handle auth (Helix → Credentials)
      - Configure google-genai Client with correct base_url, headers
      - Expose a simple `generate(prompt: str) -> str` method
    """

    def __init__(self, project_id: str, base_url: str, location: str = "us-central1"):
        self.project_id = project_id
        self.base_url = base_url
        self.location = location

        credentials = Credentials(get_coin_token())

        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            http_options=HttpOptions(
                base_url=self.base_url,
                # Citi-specific header – keep or adjust as required
                headers={"x-r2d2-soeid": os.environ.get("USER", "")},
            ),
            location=self.location,
            credentials=credentials,
        )

    def generate(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.0,
    ) -> str:
        """
        Send a plain-text prompt to the model and return the text response.
        """

        response = self.client.models.generate_content(
            model=model,
            config=GenerateContentConfig(
                temperature=temperature,
            ),
            contents=prompt,
        )

        # google-genai exposes .text for the combined output
        return response.text


# ------------------------------------------------------------------
# Global instance – configure via env vars or hardcode for POC
# ------------------------------------------------------------------

PROJECT_ID = os.getenv("R2D2_PROJECT_ID", "prj-gen-ai-9571")
BASE_URL = os.getenv(
    "R2D2_BASE_URL",
    # ⛔ IMPORTANT: replace this placeholder with your actual R2D2 endpoint
    "https://<your-r2d2-endpoint>/v1",
)
LOCATION = os.getenv("R2D2_LOCATION", "us-central1")

vertex_gen_ai = VertexGenAI(
    project_id=PROJECT_ID,
    base_url=BASE_URL,
    location=LOCATION,
)
