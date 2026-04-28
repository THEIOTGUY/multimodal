# from .clip_map import CLIPMap
# from .clip_sparse_map import CLIPSparseMap
# from .concept_fusion_3d import ConceptFusion3D
# from .gradcam_map import GradCAMMap
# from .gtmap import GTMap
# from .vlmap import VLMap
from .vlmap import VLMap
from .area_map import AreaMap
try:
    # VisualMap drags in third_party.SuperGluePretrainedNetwork + hloc; needed
    # only for image-query localization. Make optional so the mapping / sound
    # / top-down pipeline runs without those submodules.
    from .visual_map import VisualMap
except ModuleNotFoundError:
    VisualMap = None
from .sound_map import SoundMap
from .vlmap_multi_floor import VLMapMultiFloor

__all__ = ["VLMap", "VLMapMultiFloor", "AreaMap", "VisualMap", "SoundMap"]
# __all__ = ["CLIPMap", "GradCAMMap", "GTMap", "VLMap", "SoundMap", "CLIPSparseMap", "VLMap3D", "ConceptFusion3D"]
