"""
Live Data Technologies API Authentication Module

Step 1: API Authentication
A script that successfully retrieves a Bearer token from Live Data Technologies.

Author: Coastal Labor-Resilience Engine Team
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Configuration
LIVE_DATA_API_BASE_URL = "https://api.livedata.io/v1"


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""
    pass


class TokenManager:
    """
    Manages authentication tokens for Live Data Technologies API.
    
    Handles token retrieval, caching, and automatic refresh when expired.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = LIVE_DATA_API_BASE_URL
    ):
        """
        Initialize the Token Manager.
        
        Args:
            api_key: API key for authentication. Falls back to LIVE_DATA_API_KEY env var.
            api_secret: API secret for authentication. Falls back to LIVE_DATA_API_SECRET env var.
            base_url: Base URL for the API.
        """
        self.api_key = api_key or os.getenv("LIVE_DATA_API_KEY")
        self.api_secret = api_secret or os.getenv("LIVE_DATA_API_SECRET")
        self.base_url = base_url
        
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        self._validate_credentials()
    
    def _validate_credentials(self) -> None:
        """Validate that API credentials are configured."""
        if not self.api_key:
            raise AuthenticationError(
                "API key not configured. "
                "Set LIVE_DATA_API_KEY environment variable or pass api_key parameter."
            )
        if not self.api_secret:
            raise AuthenticationError(
                "API secret not configured. "
                "Set LIVE_DATA_API_SECRET environment variable or pass api_secret parameter."
            )
    
    def authenticate(self) -> Dict[str, str]:
        """
        Authenticate with Live Data Technologies API.
        
        Sends credentials to the auth endpoint and retrieves access tokens.
        
        Returns:
            Dict containing token information:
            {
                'access_token': str,
                'refresh_token': str (if provided),
                'token_type': str,
                'expires_in': int
            }
            
        Raises:
            AuthenticationError: If authentication fails.
        """
        logger.info("Authenticating with Live Data Technologies API...")
        
        auth_url = f"{self.base_url}/auth/token"
        
        payload = {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "grant_type": "client_credentials"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(
                auth_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Log response status for debugging
            logger.debug(f"Auth response status: {response.status_code}")
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API credentials.")
            
            if response.status_code == 403:
                raise AuthenticationError("API access forbidden. Check your subscription status.")
            
            response.raise_for_status()
            
            token_data = response.json()
            
            # Store tokens
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            
            # Calculate expiry time
            expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
            
            logger.info("Successfully authenticated.")
            logger.info(f"Token expires in {expires_in} seconds.")
            
            return {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "token_type": token_data.get("token_type", "Bearer"),
                "expires_in": expires_in
            }
            
        except requests.exceptions.ConnectionError:
            raise AuthenticationError("Failed to connect to API. Check network connection.")
        except requests.exceptions.Timeout:
            raise AuthenticationError("Authentication request timed out.")
        except requests.exceptions.HTTPError as e:
            raise AuthenticationError(f"HTTP error during authentication: {e}")
        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def refresh_access_token(self) -> str:
        """
        Refresh the access token using the refresh token.
        
        Returns:
            str: New access token.
            
        Raises:
            AuthenticationError: If refresh fails or no refresh token available.
        """
        if not self._refresh_token:
            logger.info("No refresh token available. Performing full authentication.")
            return self.authenticate()["access_token"]
        
        logger.info("Refreshing access token...")
        
        refresh_url = f"{self.base_url}/auth/refresh"
        
        payload = {
            "refresh_token": self._refresh_token
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                refresh_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            
            logger.info("Successfully refreshed access token.")
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Token refresh failed: {e}. Attempting full re-authentication.")
            return self.authenticate()["access_token"]
    
    @property
    def access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            str: Valid Bearer token.
        """
        if not self._access_token:
            self.authenticate()
        elif self._token_expiry and datetime.now() >= self._token_expiry:
            self.refresh_access_token()
        
        return self._access_token
    
    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with a valid token."""
        if not self._access_token:
            return False
        if self._token_expiry and datetime.now() >= self._token_expiry:
            return False
        return True
    
    def get_auth_header(self) -> Dict[str, str]:
        """
        Get authorization header for API requests.
        
        Returns:
            Dict with Authorization header ready to be used in requests.
        """
        return {
            "Authorization": f"Bearer {self.access_token}"
        }


def get_bearer_token(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> str:
    """
    Simple function to retrieve a Bearer token from Live Data Technologies.
    
    This is a convenience function for quick token retrieval without
    maintaining a TokenManager instance.
    
    Args:
        api_key: Optional API key. Falls back to environment variable.
        api_secret: Optional API secret. Falls back to environment variable.
        
    Returns:
        str: Bearer token for API authentication.
        
    Example:
        >>> token = get_bearer_token()
        >>> headers = {"Authorization": f"Bearer {token}"}
    """
    manager = TokenManager(api_key=api_key, api_secret=api_secret)
    return manager.access_token


# ============================================================================
# Main - Test Authentication
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Live Data Technologies - API Authentication Test")
    print("=" * 60)
    
    try:
        # Create token manager
        token_manager = TokenManager()
        
        # Authenticate
        token_info = token_manager.authenticate()
        
        print("\n✅ Authentication Successful!")
        print(f"   Token Type: {token_info['token_type']}")
        print(f"   Expires In: {token_info['expires_in']} seconds")
        print(f"   Token (first 20 chars): {token_info['access_token'][:20]}...")
        
        # Show auth header
        auth_header = token_manager.get_auth_header()
        print(f"\n   Auth Header: {list(auth_header.keys())[0]}: Bearer ...")
        
    except AuthenticationError as e:
        print(f"\n❌ Authentication Failed: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("  - LIVE_DATA_API_KEY")
        print("  - LIVE_DATA_API_SECRET")
        print("\nOr create a .env file with these values.")
