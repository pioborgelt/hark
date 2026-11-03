"""Fixed midpoint of matched own endpoints; no optimizer or invented loss recipe."""
import copy
from .weight_average_r174 import average_state_dict
from .model import checkpoint_parameters
from .parent_training_recipe import recipe_from_checkpoint


def midpoint_checkpoint(left, right, left_sha256, right_sha256, plan_sha256):
    required=['vocab','hidden','bins','causal_layers','dropout','manifest_hash','feature_version',
              'text_version','parameters','alignment_weight','parent_checkpoint_sha256',
              'base_manifest_sha256','expanded_music_sha256','room_feature_sha256',
              'timed_annotation_source_sha256','ctc_segment','negative_music_frames',
              'background_pool_size','negative_music_pool_size','new_train_rows']
    for key in required:
        if key not in left or left[key]!=right.get(key):raise ValueError('Unmatched source contract: '+key)
    for ck in [left,right]:
        if ck.get('step')!=1000 or ck.get('model_kind'):raise ValueError('Complete own matched1000-step base endpoints required')
        if ck.get('posterior_temperature',1.)!=1. or ck.get('global_nonblank_head_offset',0.)!=0.:
            raise ValueError('Use raw source heads before calibration')
    recipes=[recipe_from_checkpoint(ck) for ck in [left,right]]
    if [r['music_peak_weight'] for r in recipes]!=[0.,.5]:raise ValueError('Expected the registered music-loss comparison')
    if {k:v for k,v in recipes[0].items() if k!='music_peak_weight'}!={k:v for k,v in recipes[1].items() if k!='music_peak_weight'}:
        raise ValueError('Only source music-loss weight may differ')
    averaged=average_state_dict(checkpoint_parameters(left),checkpoint_parameters(right),.5)
    # Architecture, frontend, transcript/annotation provenance and original
    # lineage stay explicit. No source SGD history is attributed to this step0.
    ck={key:copy.deepcopy(left[key]) for key in required}
    ck.update(model=averaged,step=0,epoch=0,best=None,
        args={'max_steps':0,'smoke_rows':0,'batch':96,'blank_loss_weight':left['args']['blank_loss_weight'],
              'out':'artifacts/r196-matched-midpoint','execution_kind':'raw_parameter_derivation'},
        parent_checkpoint_sha256=left_sha256,
        initialization='Fixed half/half own matched raw endpoints; no new SGD',
        execution_kind='raw_parameter_derivation',new_sgd_steps=0,
        single_sgd_training_recipe_applicable=False,
        source_training_args={'left':copy.deepcopy(left['args']),'right':copy.deepcopy(right['args'])},
        source_training_recipes={'left':recipes[0],'right':recipes[1]},
        parameter_derivation={'left_sha256':left_sha256,'right_sha256':right_sha256,
            'right_weight':.5,'source_local_steps':[1000,1000],'new_sgd_steps':0,
            'training_plan_sha256':plan_sha256,
            'not_equivalent_to_training_at_averaged_music_loss_weight':True})
    return ck
