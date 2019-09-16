import numpy as np
from pathlib import Path

SOURCE_TYPE = {
    'ID': np.uint32,
    'assigned_date': np.uint16,
    'assigned_to_group': str,
    'case_type': str,
    'category': str,
    'channel rt': str,
    'create_date': np.uint16,
    'details': str,
    'first_assigned_group': str,
    'impact': np.uint8,
    #     'isriver': np.uint8,
    'item': str,
    'root_cause': str,  # dependent variable
    'short_description': str,
    'site': str,
    'site_building_type': str,
    'site_city': str,
    'site_country': str,
    'site_loc_bldg_cd': str,
    'site_region': str,
    'site_state': str,
    'type': str,
}

# Context variables
CTX_DF = 'df'

# Tokens
KIOSK = ' xxkiosk '
MAC = ' xxmac '
TRACKING = ' xxtracking '
SERIAL = ' xxserial '
ASSET = ' xxasset '
RMA = ' xxrma '
IP = ' xxip '
TT = ' xxtt '

AMZ_EMAIL = ' xxamzemail '
OTHER_EMAIL = ' xxotheremail '
ALIAS = ' xxalias '
IMEI = ' xximei '
SIMNUM = ' xxsimnum '
PHONE = ' xxphone '
EC2_INSTANCE = ' xxectwoinstance '

# Don't add the ones below to ALL_FEATURES
OTHER_AMZ_LINK = ' xxotheramzlink '
OTHER_EXT_LINK = ' xxotherextlink '
WFSS_WORKFLOW = ' xxwfssworkflow '
BUILDING = ' xxbuilding '
HOST = ' xxhost '
UUID = ' xxuuid '

ALL_FEATURES = [
    KIOSK, MAC, TRACKING, SERIAL, ASSET, RMA, IP, TT,
    AMZ_EMAIL, OTHER_EMAIL, ALIAS, IMEI, EC2_INSTANCE, SIMNUM, PHONE
]

DATASET_SAMPLE = 'sample'
DATASET_TRAIN = 'training'
DATASET_TEST = 'public_test_features'

PATH = Path('../../../../../data')

DATASETS = {
    DATASET_SAMPLE: {'is_training': True, 'write_csv': True, 'csv_cols': None},
    # DATASET_SAMPLE: {'is_training': True, 'write_csv': True, 'csv_cols': ['details']},
    DATASET_TRAIN: {'is_training': True, 'write_csv': False},
    DATASET_TEST: {'is_training': False, 'write_csv': False, },
}
