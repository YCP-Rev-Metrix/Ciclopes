from fastapi import APIRouter

router = APIRouter(
    prefix="/fourdbody",
    tags=["fourDbody"],
)

# Add endpoints here, e.g.
# @router.get("/health")
# async def health():
#     return {"ok": True}
