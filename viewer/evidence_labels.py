
from __future__ import annotations
from viewer.layer_controller import EvidenceProfile, resolve_evidence_profile
from viewer.project_model import ScanEntry, ScanLayer
EVIDENCE_ITEMS: tuple[tuple[str, str, str], ...] = ((EvidenceProfile.NATIVE_RGB.value, 'KAYNAK', 'Yerlesim gorunumu: her gorunur taramanin kendi dogrulanmis RGB kaynagi (ham nokta, uygulama yuzeyi veya mesh) cizilir. Duzeltilmis katman secilmez; secili taramanin gercek kaynagi Ozellikler panelinde yazilir.'), (EvidenceProfile.CORRECTED_RGB.value, 'DUZELTILMIS', 'Ham depth/conf karelerinden uretilen geometri; kayitli menzil modeli uygulanmistir. Menzil kesilmez, nokta silinmez.'), (EvidenceProfile.RAW_RGB.value, 'HAM', 'Ham depth/conf karelerinden uretilen temel geometri. Menzil modeli uygulanmamistir; menzil kesilmez, nokta silinmez.'), (EvidenceProfile.APP_FUSED.value, 'UYGULAMA', 'Tarama uygulamasinin kendi birlestirdigi yuzey. Bu kaynak CCA donusumu veya duzeltilmis geometri degildir.'), (EvidenceProfile.REVIEW_CORE.value, 'ARSIV', 'Eski inceleme kopyasi. Yalnizca baska geometri kaynagi yoksa kullanilir; duzeltilmis veya ham sonucu temsil etmez.'))
_ITEMS_BY_KEY = {key: (label, description) for key, label, description in EVIDENCE_ITEMS}
_ITEMS_BY_KEY[EvidenceProfile.RAW_CONF.value] = ('HAM', 'Eski kayit anahtari; ayni ham geometri CONFIDENCE renkle gosterilir.')
_CORRECTED_RAW_ROLES = frozenset({'CORRECTED_RAW_RGB', 'CORRECTED_RAW_CONFIDENCE', 'CORRECTED_RAW_RGB_PRIMARY'})
_RAW_FRAME_ROLES = frozenset({'RAW_FRAME_RGB_EVIDENCE', 'RAW_FRAME_CONFIDENCE_EVIDENCE', 'RAW_FRAME_EVIDENCE'})
_APP_FUSED_ROLES = frozenset({'PRIMARY_APP_FUSED_SURFACE', 'APP_FUSED_SURFACE'})
_REVIEW_ROLES = frozenset({'VIEWER_CORE', 'VIEWER_BODY', 'VIEWER_SHELL'})

def evidence_label(profile: EvidenceProfile | str) -> str:
    key = resolve_evidence_profile(profile).value
    return _ITEMS_BY_KEY[key][0]

def evidence_provenance(layer: ScanLayer) -> str:
    if layer.role in _CORRECTED_RAW_ROLES:
        correction = layer.metadata.get('depth_correction')
        if isinstance(correction, dict) and isinstance(correction.get('kind'), str):
            return f"CCA tarafindan ham depth/conf karelerinden birlestirilmis bulut. Sensor yonu ve kamera olcegi duzeltildi. Menzil duzeltmesi uygulandi: {correction['kind']}; menzil kesilmedi, nokta silinmedi."
        return 'CCA tarafindan ham depth/conf karelerinden birlestirilmis bulut. Sensor yonu ve kamera olcegi duzeltildi. Gurultu/bias duzeltmesi uygulanmadi.'
    if layer.role in _RAW_FRAME_ROLES:
        return 'Ham depth/conf karelerinden turetilmis temel CCA geometri bulutu: K olcekleme + geri izdusum + CAD Z-yukari.'
    if layer.role in _APP_FUSED_ROLES:
        return 'Tarama uygulamasinin birlestirdigi nokta bulutu.'
    if layer.role == 'IMPORTED_SOURCE_POINTS':
        return 'Disaridan alinan PLY bulutu; uretim yontemi dogrulanmamis.'
    if layer.role == 'IMPORTED_SOURCE_MESH':
        return 'Disaridan alinan legacy mesh; kaynak ekseni ve olcum dogrulugu belirtilmemis, plan uzerinde elle yerlestirilir.'
    if layer.role in _REVIEW_ROLES:
        return 'Inceleme icin ayrilmis katman; ilk uretim yontemi belirtilmemis.'
    return 'Kaynak katmanin uretim yontemi belirtilmemis.'

def evidence_tooltip(profile: EvidenceProfile | str, scan: ScanEntry | None=None, source_layer_key: str | None=None) -> str:
    key = resolve_evidence_profile(profile).value
    description = _ITEMS_BY_KEY[key][1]
    if scan is None:
        return description
    if source_layer_key is None:
        return description + '\nSecili tarama: kaynak katmani belirtilmedi.'
    layer = scan.layer(source_layer_key)
    if layer is None:
        return description + '\nSecili taramada bu kaynak katmani kayitli degil.'
    return description + '\nSecili kaynak: ' + evidence_provenance(layer)
__all__ = ['EVIDENCE_ITEMS', 'evidence_label', 'evidence_provenance', 'evidence_tooltip']
