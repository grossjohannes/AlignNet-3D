import json
import os

dataset_dir = '/home/gross/data'

for dataset in ['SynthCars', 'SynthCarsPersons', 'Synth20', 'Synth20others', 'KITTITrackletsCars', 'KITTITrackletsCarsPersons', 'KITTITrackletsCarsHard', 'KITTITrackletsCarsPersonsHard']:
    for filename, icp_variant, refine in [('o3_p2p', 'p2point', False), ('o3_gicp', 'o3_gicp', False), ('o3_gicp_p2p', 'o3_gicp', True), ('o3_gicp_fast', 'o3_gicp_fast', False), ('o3_gicp_fast_p2p', 'o3_gicp_fast', True)]:
        cfg = {  #
            'data': {
                'basepath': os.path.join(dataset_dir, dataset)
            },
            'evaluation': {
                'special': {
                    'mode': 'icp',
                    'icp': {
                        'variant': icp_variant,
                        'with_constraint': True,
                    }
                }
            }
        }
        if refine:
            cfg['evaluation']['special']['icp']['refine'] = 'p2p'

        with open(os.path.join(os.path.dirname(__file__), 'configs', f'icp_{dataset}_{filename}.json'), 'w') as f:
            json.dump(cfg, f, indent=4)
