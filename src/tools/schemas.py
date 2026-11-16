from pydantic import BaseModel, ConfigDict, Field

class GetOrderStatusInput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    order_id: str = Field(
        min_length=4,
        max_length=30,
        description="Unique customer order ID"
    )