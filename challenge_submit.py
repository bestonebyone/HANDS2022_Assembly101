import numpy as np
import json

def db_size(set_name):
    """ our input is a video and we predict per 60 frames,
        i.e., we predict at the [0,60,120,...,342*60] frames.
    """
    if set_name == '100446':
        return 343  
    else:
        assert 0, 'Invalid choice.'

def dump(pred_out_path, xyz_pred_list):
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
            ], fo)
    print('Dumped %d joints to %s' % (len(xyz_pred_list), pred_out_path))


def pred_template(img=None, camera_intrinsic=None, camera_extrinsic=None):
    """ Predict joints and vertices from a given sample.
        img: RGB images or egocentric images.
        camera_intrinsic: camera intrinsic matrix.
        camera_extrinsic: camera extrinsic matrix.
    """
    # TODO: Put your algorithm here, which computes 3D joint coordinates 
    xyz = np.zeros((42, 3))  # 3D coordinates of two hands for the input image
    return xyz

set_name = '100446'

xyz_pred = list()
for i in range(db_size(set_name)):
    pose = pred_template()
    xyz_pred.append(pose)

dump('./pred_'+set_name+'.json', xyz_pred)