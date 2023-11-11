import numpy as np
import nibabel as nib
from nibabel.gifti import gifti
import glob
from xml.etree import ElementTree as et


class Logging:
    """
    Inputs:
    - subj_dir: prefix of the directory for the subject
    - message: information to be logged
    """
    def __init__(self, subj_dir):
        self.subj_dir = subj_dir
        # clean the previous log
        with open(subj_dir+'.log', 'w') as f:
            f.write('')
    def info(self, message, end='\n'):
        print(message, end=end)
        with open(self.subj_dir+'.log', 'a') as f:
            f.write(message+end)
    
    
def save_numpy_to_nifti(img, affine, save_dir):
    """
    Save numpy array to nifti image (.nii.gz).
    
    Inputs:
    - img: image to be save, (D1,D2,D3) numpy.array 
    - affine: affine matrix, (4,4) numpy.array
    - save_dir: directory for saving, string
    """
    img_nib = nib.Nifti1Image(img, affine)
    img_nib.header['xyzt_units'] = 2
    nib.save(img_nib, save_dir)


def save_gifti_surface(
    vert, face, save_dir, surf_hemi='left', surf_type='wm'):
    """
    Save gifti surface (.surf.gii).
    
    Inputs:
    - vert: mesh vertices, (|V|,3) numpy.float32
    - face: mesh faces, (|F|,3) numpy.int32
    - save_dir: directory for saving, string
    - surf_hemi: ['left', 'right']
    - surf_type: ['wm', 'pial', 'midthickness',
                  'inflated', 'vinflated', 'sphere']
    """

    # convert args to gifti header
    if surf_hemi == 'left':
        _surf_hemi = 'CortexLeft'
    elif surf_hemi == 'right':
        _surf_hemi = 'CortexRight'
        
    if surf_type == 'wm':
        _surf_type = 'GrayWhite'
        _geo_type = 'Anatomical'
    elif surf_type == 'pial':
        _surf_type = 'Pial'
        _geo_type = 'Anatomical'
    elif surf_type == 'midthickness':
        _surf_type = 'MidThickness'
        _geo_type = 'Anatomical'
    elif surf_type == 'inflated':
        _surf_type = 'MidThickness'
        _geo_type = 'Inflated'
    elif surf_type == 'vinflated':
        _surf_type = 'MidThickness'
        _geo_type = 'VeryInflated'
    elif surf_type == 'sphere':
        _surf_type = 'MidThickness'
        _geo_type = 'Spherical'
        
    # meta data
    vert_meta_dict = {'<![CDATA[AnatomicalStructurePrimary]]>':
                      '<![CDATA['+_surf_hemi+']]>',
                      '<![CDATA[AnatomicalStructureSecondary]]>':
                      '<![CDATA['+_surf_type+']]>',
                      '<![CDATA[GeometricType]]>':
                      '<![CDATA['+_geo_type+']]>',
                      '<![CDATA[Name]]>': '<![CDATA[#1]]>'}
    face_meta_dict = {'<![CDATA[Name]]>': '<![CDATA[#2]]>'}

    vert_meta = gifti.GiftiMetaData(vert_meta_dict)
    face_meta = gifti.GiftiMetaData(face_meta_dict)

    # create gifti data
    gii_surf = gifti.GiftiImage()
    gii_surf_vert = gifti.GiftiDataArray(
        vert.astype(np.float32), intent='pointset', meta=vert_meta)
    gii_surf_face = gifti.GiftiDataArray(
        face.astype(np.int32), intent='triangle', meta=face_meta)
    gii_surf.add_gifti_data_array(gii_surf_vert)
    gii_surf.add_gifti_data_array(gii_surf_face)
    
    # save gifti xml file (.gii)
    gii_file = gii_surf.to_xml().decode("utf-8");
    gii_file = gii_file.replace("&lt;", "<");
    gii_file = gii_file.replace("&gt;", ">");
    
    with open(save_dir, 'wb') as f: 
        f.write(gii_file.encode("utf-8"))
    # nib.save(gii_surf, save_dir)


    
def save_gifti_metric(
    metric, save_dir, surf_hemi='left', metric_type='curv'):
    """
    Save gifti metric (.shape.gii).
    
    Inputs:
    - metric: mesh metric, (|V|) numpy.float32
    - save_dir: directory for saving, string
    - surf_hemi: ['left', 'right']
    - metric_type: ['thickness', 'curv', 'sulc', 'roi',
                    'myelinmap', 'smoothed_myelinmap']
    """
    
    # convert args to gifti header
    if surf_hemi == 'left':
        _surf_hemi = 'CortexLeft'
    elif surf_hemi == 'right':
        _surf_hemi = 'CortexRight'
    
    # set meta data
    if metric_type == 'thickness':
        _metric_type = 'Thickness'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 7.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'videen_style'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'false'
        DisplayNegativeData = 'false'
        
    elif metric_type == 'curv':
        _metric_type = 'Curvature'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 2.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'PSYCH-NO-NONE'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'true'
        DisplayNegativeData = 'true'
        
    elif metric_type == 'sulc':
        _metric_type = 'SulcalDepth'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 2.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'ROY-BIG-BL'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'true'
        DisplayNegativeData = 'true'
        
    elif metric_type == 'roi':
        _metric_type = 'ROI'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 2.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'POS_NEG_ZERO'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'false'
        DisplayNegativeData = 'false'
        
    elif metric_type == 'myelinmap':
        _metric_type = 'MyelinMap'
        ScaleMode = 'MODE_USER_SCALE'
        # ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 4.000000 96.000000'
        UserScaleValues = '0.000000 0.000000 1.000000 1.700000'
        PaletteName = 'videen_style'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'false'
        DisplayNegativeData = 'false'
        
    elif metric_type == 'smoothed_myelinmap':
        _metric_type = 'MyelinMap'
        ScaleMode = 'MODE_USER_SCALE'
        # ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 7.000000 98.000000'
        UserScaleValues = '0.000000 0.000000 1.000000 1.700000'
        PaletteName = 'videen_style'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'false'
        DisplayNegativeData = 'false'
    # meta data for color map
    cmap_meta = \
    '<PaletteColorMapping Version="1">\n'+\
    '<ScaleMode>'+ScaleMode+'</ScaleMode>\n'+\
    '<AutoScalePercentageValues>'+AutoScalePercentageValues+'</AutoScalePercentageValues>\n'+\
    '<AutoScaleAbsolutePercentageValues>2.000000 98.000000</AutoScaleAbsolutePercentageValues>\n'+\
    '<UserScaleValues>'+UserScaleValues+'</UserScaleValues>\n'+\
    '<PaletteName>'+PaletteName+'</PaletteName>\n'+\
    '<InterpolatePalette>true</InterpolatePalette>\n'+\
    '<DisplayPositiveData>'+DisplayPositiveData+'</DisplayPositiveData>\n'+\
    '<DisplayZeroData>'+DisplayZeroData+'</DisplayZeroData>\n'+\
    '<DisplayNegativeData>'+DisplayNegativeData+'</DisplayNegativeData>\n'+\
    '<ThresholdTest>THRESHOLD_TEST_SHOW_OUTSIDE</ThresholdTest>\n'+\
    '<ThresholdType>THRESHOLD_TYPE_OFF</ThresholdType>\n'+\
    '<ThresholdFailureInGreen>false</ThresholdFailureInGreen>\n'+\
    '<ThresholdNormalValues>-1.000000 1.000000</ThresholdNormalValues>\n'+\
    '<ThresholdMappedValues>-1.000000 1.000000</ThresholdMappedValues>\n'+\
    '<ThresholdMappedAvgAreaValues>-1.000000 1.000000</ThresholdMappedAvgAreaValues>\n'+\
    '<ThresholdDataName></ThresholdDataName>\n'+\
    '<ThresholdRangeMode>PALETTE_THRESHOLD_RANGE_MODE_MAP</ThresholdRangeMode>\n'+\
    '<ThresholdLowHighLinked>false</ThresholdLowHighLinked>\n'+\
    '<NumericFormatMode>AUTO</NumericFormatMode>\n'+\
    '<PrecisionDigits>2</PrecisionDigits>\n'+\
    '<NumericSubivisions>0</NumericSubivisions>\n'+\
    '<ColorBarValuesMode>DATA</ColorBarValuesMode>\n'+\
    '<ShowTickMarksSelected>false</ShowTickMarksSelected>'
    
    metric_meta_dict = {'<![CDATA[Name]]>': '<![CDATA['+_metric_type+']]>',
                        '<![CDATA[PaletteColorMapping]]>':
                        '<![CDATA['+cmap_meta+']]>'}
    metric_meta = gifti.GiftiMetaData(metric_meta_dict)
        
    # meta data
    metric = metric.astype(np.float32)
    gii_meta_dict = {'<![CDATA[AnatomicalStructurePrimary]]>':
                     '<![CDATA['+_surf_hemi+']]>'}
    gii_meta = gifti.GiftiMetaData(gii_meta_dict)

    # new gii image
    gii_metric = gifti.GiftiImage(meta=gii_meta)
    gii_metric_arr = gifti.GiftiDataArray(metric, intent='normal', meta=metric_meta)
    gii_metric.add_gifti_data_array(gii_metric_arr)
    
    # save gifti xml file (.gii)
    gii_file = gii_metric.to_xml().decode("utf-8");
    gii_file = gii_file.replace("&lt;", "<");
    gii_file = gii_file.replace("&gt;", ">");

    with open(save_dir, 'wb') as f: 
        f.write(gii_file.encode("utf-8"))
    # nib.save(gii_metric, save_dir)

    
def create_wb_spec(subj_dir):
    """
    Create workbench specification file (.spec) for visulzaion
    
    Inputs:
    - subj_dir: directory of the subject.
    """
    # create header
    caret = et.Element('CaretSpecFile', Version='1.0')
    et.SubElement(caret, 'MetaData').text = ' '

    # add surface/metric files to .spec
    for surf_hemi in ['left','right']:
        for file_type in ['surf', 'shape']:
            if surf_hemi == 'left':
                structure = 'CortexLeft'
            elif surf_hemi == 'right':
                structure = 'CortexRight'
            if file_type == 'surf':
                data_file_type = 'SURFACE'
            elif file_type == 'shape':
                data_file_type = 'METRIC'
            file_list = glob.glob(
                subj_dir+'_hemi-'+surf_hemi+'*.'+file_type+'.gii')
            # add to xml tree
            for file_name in file_list:
                et.SubElement(
                    caret, 'DataFile',
                    Structure=structure,
                    DataFileType=data_file_type,
                    Selected='true').text = file_name.split('/')[-1]

    # add volume files to .spec
    file_name = subj_dir+'_T2w_restore_brain.nii.gz'
    # add to xml tree
    et.SubElement(
        caret, 'DataFile',
        Structure='Invalid',
        DataFileType='VOLUME',
        Selected='true').text = file_name.split('/')[-1]

    # save .spec file
    spec = et.ElementTree(caret)
    et.indent(spec)
    spec.write(
        subj_dir + '_wb.spec',
        encoding='UTF-8',
        xml_declaration=True)
    