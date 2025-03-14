import os
import xml.etree.ElementTree as ET
from io import StringIO
import numpy as np
import glob
import warnings

class Camera_Loader:
    def __init__(self,dir):
        self.load_dir = dir
        self.key_element = ['Camera_Matrix', 'Distortion_Coefficients','cam_translation','cam_rotation']
        self.camera_number = 18
        self.camera_info = {camera_id:{} for camera_id in range(18)}


        for camera_id in range(self.camera_number):
            file_name = 'cam' + str(camera_id).zfill(2) + '.xml'
            self.data = {}
            self._load_data(os.path.join(dir,file_name))
            if not self.data:
                warnings.warn('Not find xml under provided directory, please input another directory')
            self.camera_info[camera_id] = self.data.copy()


    def _load_data(self,xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for element in self.key_element:
            r,c,dt,data = root.find(element)
            temp = (data.text).replace('\n','')
            self.data[element] = np.genfromtxt(StringIO(' '.join(temp.split())),delimiter=' ').reshape((int(r.text),int(c.text)))

    def __str__(self):
        return str(self.camera_info)

    def __getitem__(self, camera_id):
        #camera_id from 0 to 17
        return self.camera_info[camera_id]




if __name__ == '__main__':
    camera_load = Camera_Loader('XGaze_utils/camera_parameters')
    print(camera_load[1])
    #print(camera_load[0])
    # tree = ET.parse('camera_parameters/cam00.xml')
    # root = tree.getroot()
    # dis_coef = root.find('Distortion_Coefficients')
    #
    # dis_coef = root.find('cam_translation')
