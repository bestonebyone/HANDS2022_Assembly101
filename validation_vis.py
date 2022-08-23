import numpy as np
import cv2,json,os

# set the validation path
validation_folder = './validation/'

# set the camera
camera_number = 2

# load camera name
cam_name_list = ["C10119_rgb","C10404_rgb","C10115_rgb","C10390_rgb",
                "C10395_rgb","C10119_rgb", "C10118_rgb","C10379_rgb",
                "HMC_21110305_mono10bit","HMC_21176623_mono10bit",
                "HMC_21176875_mono10bit","HMC_21179183_mono10bit"]

cam_name = cam_name_list[camera_number]

# load the path
file_name = 'nusar-2021_action_both_9042-c09c_9042_user_id_2021-02-17_102611/'
cams_path = validation_folder+'camera_metadata/'+file_name
pose_path = validation_folder+'annotation/'+file_name
if cam_name[0] == 'C':
    video_path = validation_folder + 'recordings/' + file_name + cam_name +'.mp4'
else:
    video_path = validation_folder + 'camera_metadata/' + file_name + cam_name +'_undistort.mp4'


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def get_cams_matrix(cam_name, idx, path = cams_path):
    # load the camera info from calib.txt in camera_metadata

    camera_intrinsic_dict = {}
    camera_intrinsic_dict['C10095_rgb'] = np.array([[132.95890021, 0, 313.04244009], [0, 132.91978720, 233.64238351], [0, 0, 1]])
    camera_intrinsic_dict['C10404_rgb'] = np.array([[1210.55521582, 0, 976.53667731], [0, 1210.40163069, 522.81143699], [0, 0, 1]])
    camera_intrinsic_dict['C10115_rgb'] = np.array([[1204.96465374, 0, 944.46194031], [0, 1205.23115124, 517.69094104], [0, 0, 1]])
    camera_intrinsic_dict['C10390_rgb'] = np.array([[1204.14322366, 0, 943.69011465], [0, 1204.45991241, 527.95581067], [0, 0, 1]])
    camera_intrinsic_dict['C10395_rgb'] = np.array([[1200.69422082, 0, 948.46950579], [0, 1201.13323568, 521.22254409], [0, 0, 1]])
    camera_intrinsic_dict['C10119_rgb'] = np.array([[1203.10622333, 0, 955.08791383], [0, 1203.70582507, 517.62506005], [0, 0, 1]])
    camera_intrinsic_dict['C10118_rgb'] = np.array([[1208.01889094, 0, 954.08070929], [0, 1208.54113321, 520.18891200], [0, 0, 1]])
    camera_intrinsic_dict['C10379_rgb'] = np.array([[1209.65833376, 0, 951.99052683], [0, 1208.57110945, 527.69580162], [0, 0, 1]])

    camera_extrinsic_dict = {}
    camera_extrinsic_dict['C10095_rgb'] = np.array([[0.68514128, -0.00273721, -0.72840506, 38.59722542 ], [0.49687025, -0.72946979, 0.47009975, -21.02694914 ], [-0.53263626, -0.68400755, -0.49842983, 1085.30222663 ]])
    camera_extrinsic_dict['C10404_rgb'] = np.array([[-0.99118988, -0.06871655, -0.11322838, 209.26719124 ], [0.13224976, -0.56029958, -0.81766398, 18.80784972 ], [-0.00725476, -0.82543468, 0.56445102, 1182.52557541 ]])
    camera_extrinsic_dict['C10115_rgb'] = np.array([[0.01104671, 0.00327979, -0.99993360, 51.74820037 ], [0.99773016, -0.06646651, 0.01080436, -9.88618103 ], [-0.06642666, -0.99778327, -0.00400658, 1211.78160277 ]])
    camera_extrinsic_dict['C10390_rgb'] = np.array([[0.99608474, -0.07800602, 0.04159620, -54.86931156 ], [-0.08173035, -0.63324515, 0.76962376, -30.52390696 ], [-0.03369470, -0.77001015, -0.63714129, 1315.81977655 ]])
    camera_extrinsic_dict['C10395_rgb'] = np.array([[0.97759094, 0.02662489, 0.20882307, -67.54198005 ], [0.09198171, -0.94629160, -0.30995416, 328.52827366], [0.18935502, 0.32221628, -0.92753509, 964.82675589 ]])
    camera_extrinsic_dict['C10119_rgb'] = np.array([[-0.60227116, 0.08881533, -0.79333554, 79.28260545 ], [0.50374630, -0.72865863, -0.46400030, 122.72197183 ], [-0.61928113, -0.67909384, 0.39410968, 948.11826525 ]])
    camera_extrinsic_dict['C10118_rgb'] = np.array([[0.05392442, -0.01103973, -0.99848399, 113.09495174 ], [-0.02081259, -0.99973408, 0.00992954, 190.90478165 ], [-0.99832810, 0.02024560, -0.05413984, 660.20757522 ]])
    camera_extrinsic_dict['C10379_rgb'] = np.array([[-0.97657446, -0.04608225, 0.21018744, 133.03327790 ], [0.10553308, -0.95382633, 0.28120830, 235.12652864 ], [0.18752360, 0.29680257, 0.93634563, 775.42434630 ]])

    if cam_name[0] == 'C':
        return camera_intrinsic_dict[cam_name],camera_extrinsic_dict[cam_name]
    else:
        cams_matrix_path = path + cam_name +'_undistort.json'
        cam_matrix = json_load(cams_matrix_path)

        intrinsic = np.array(cam_matrix['intrinsics'])
        extrinsic = np.array(cam_matrix['extrinsics'][idx])
        return intrinsic,extrinsic

def get_pose3d(idx, path = pose_path):
    pose_path = path + 'keypoints_3D'+ "/%06d.pts" % (idx)
    if os.path.exists(pose_path):
        annotation_3d = np.genfromtxt(pose_path)
    else:
        print('Missing Joint:', idx)
        annotation_3d = np.zeros((42,3))

    hand_right = annotation_3d[:21,:3]  # 0:20 right hand
    hand_left = annotation_3d[21:,:3] # 21:41 left hand
    hand_all = annotation_3d[:,:3]
    confidence = annotation_3d[:,3] # confidence score of the annotation

    return hand_all, confidence

def projection(pose3d, camera_intrinsic, camera_extrinsic):
    camera_intrinsic = np.concatenate((camera_intrinsic, np.array([[0,0,0]]).reshape(3,1)),axis=-1)
    camera_extrinsic = np.concatenate((camera_extrinsic, np.array([0,0,0,1]).reshape(1,4)))
    pose3d = np.concatenate((pose3d,np.ones((pose3d.shape[0],1))),axis=-1)

    output = (camera_intrinsic.dot(camera_extrinsic.dot(pose3d.T))).T
    for i in range(output.shape[0]):
        output[i,:] = output[i,:]/output[i,2]
    uv = output[:,:2]
    return uv


cap=cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frames_count = int(frames_count)

for i in range(frames_count):
    ret,image = cap.read() 
    if i%60==0:
        xyz,confidence = get_pose3d(i)
        intrinsic,extrinsic = get_cams_matrix(cam_name, i)
        uv = projection(xyz,intrinsic,extrinsic).astype('int')
        for ii in range(uv.shape[0]):
            if ((uv[ii,0]<width) and (uv[ii,1]<height) and (uv[ii,1]>0) and (uv[ii,0]>0) and confidence[ii]>0.75):
                cv2.circle(image,(uv[ii,0],uv[ii,1]), 5, (255,0,0), 0)
        cv2.imshow('img',image)
        if cv2.waitKey(50) & 0xFF == ord('s'):
            break