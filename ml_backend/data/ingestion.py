"""
Data ingestion module - Fetch data from data.gov.in APIs.
"""
import logging
import httpx
import pandas as pd
from typing import Optional, Dict, Any, List

import sys
sys.path.insert(0, '..')

from config import get_settings, DATASETS

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data fetching from data.gov.in APIs."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.data_gov_base_url
        self.api_key = self.settings.data_gov_api_key
    
    async def fetch_data(
        self,
        dataset_id: str,
        limit: int = 1000,
        offset: int = 0,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch data from data.gov.in API.
        
        Args:
            dataset_id: One of 'enrolment', 'demographic', 'biometric'
            limit: Number of records to fetch (max 10000)
            offset: Starting offset for pagination
            filters: Optional filters to apply
            
        Returns:
            Dict containing success status, total count, and records
        """
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        resource_id = DATASETS[dataset_id]["resource_id"]
        
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": str(limit),
            "offset": str(offset)
        }
        
        if filters:
            params.update(filters)
        
        url = f"{self.base_url}/{resource_id}"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Fetched {data.get('count', 0)} records from {dataset_id}")
                
                return {
                    "success": True,
                    "total": data.get("total", 0),
                    "count": data.get("count", 0),
                    "records": data.get("records", [])
                }
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {dataset_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "records": []
            }
        except Exception as e:
            logger.error(f"Error fetching {dataset_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "records": []
            }
    
    async def fetch_all_pages(
        self,
        dataset_id: str,
        max_records: int = 5000,
        page_size: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch multiple pages of data and return as DataFrame.
        
        Args:
            dataset_id: Dataset to fetch
            max_records: Maximum total records to fetch
            page_size: Records per API call
            
        Returns:
            pandas DataFrame with all fetched records
        """
        all_records: List[Dict] = []
        offset = 0
        
        while len(all_records) < max_records:
            result = await self.fetch_data(
                dataset_id,
                limit=min(page_size, max_records - len(all_records)),
                offset=offset
            )
            
            if not result["success"] or not result["records"]:
                break
                
            all_records.extend(result["records"])
            offset += len(result["records"])
            
            # Check if we've fetched all available records
            if len(result["records"]) < page_size:
                break
        
        logger.info(f"Total records fetched for {dataset_id}: {len(all_records)}")
        
        if not all_records:
            return pd.DataFrame()
            
        return pd.DataFrame(all_records)
    
    def to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """Convert list of records to pandas DataFrame."""
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)


# Singleton instance - reset on module reload to pick up new config
_ingestion_instance: Optional[DataIngestion] = None


def get_data_ingestion() -> DataIngestion:
    """Get or create DataIngestion singleton."""
    global _ingestion_instance
    if _ingestion_instance is None:
        # Clear lru_cache on settings to get fresh config
        from config import get_settings
        get_settings.cache_clear()
        _ingestion_instance = DataIngestion()
    return _ingestion_instance


def reset_data_ingestion():
    """Reset the singleton to pick up new config (e.g. API key change)."""
    global _ingestion_instance
    from config import get_settings
    get_settings.cache_clear()
    _ingestion_instance = None
