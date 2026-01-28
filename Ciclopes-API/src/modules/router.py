from fastapi import APIRouter

from .fourDbody.routes import router as fourdbody_router
from .LaneBalls.routes import router as laneballs_router
from .test.routes import router as test_router

router = APIRouter()

router.include_router(test_router)
router.include_router(fourdbody_router)
router.include_router(laneballs_router)