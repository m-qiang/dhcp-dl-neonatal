import numpy as np
import ants
import scipy
import nibabel as nib
from scipy.io import loadmat


def ants_trans_to_mat(trans):
    """
    Compute the affine matrix from the Ants transformation.
    
    Inputs:
    - trans: Ants transformation (returned from Ants.registration)
    
    Returns:
    - affine: affine matrix, (4,4) numpy.array
    """
    fwd_transform = loadmat(trans['fwdtransforms'][0])
    m_matrix = fwd_transform['AffineTransform_float_3_3'][:9].reshape(3,3)# .T
    m_center = fwd_transform['fixed'][:,0]
    m_translate = fwd_transform['AffineTransform_float_3_3'][9:][:,0]
    m_offset = m_translate + m_center - m_matrix @ m_center

    # ITK affine to affine matrix
    affine = np.zeros([4,4])
    affine[:3,:3] = m_matrix
    affine[:3,-1] = -m_offset
    affine[3,:] = np.array([0,0,0,1])

    # LIP space to RAS coordinates
    affine[2,-1] = -affine[2,-1]
    affine[2,1] = -affine[2,1]
    affine[1,2] = -affine[1,2]
    affine[2,0] = -affine[2,0]
    affine[0,2] = -affine[0,2]
    return affine


def registration(
    img_move_ants,
    img_fix_ants,
    affine_fix,
    out_prefix,
    aff_metric='mattes',
    max_iter=5,
    min_dice=0.9,
    seed=10086,
    verbose=False,
):
    """
    Robust Ants rigid + affine registration from moving image to 
    fixed image. The registration is performed multiple times (max_iter)
    if the dices score < min_dice.
    
    Inputs:
    - img_move_ants: moving image to be aligned, Ants image
    - img_fix_ants: target fixed image, Ants image
    - affine_fix: affine matrix of fixed image, (4,4) numpy.array
    - out_prefix: prefix of output transformation files 
    - aff_metric: metric used for optimization ['mattes','meansquares', 'gc']
    - max_iter: maximum iterations for registration
    - min_dice: minimum required dice score after registration
    - seed: random seed for reproducing results
    - verbose: if report
    
    Returns:
    - img_align_ants: aligned image, Ants image
    - affine_mat: affine matrix after registration, (4,4) numpy.array
    - trans_rigid: transformation for rigid registration, Ants.transformation
    - trans_affine: transformation for affine registration, Ants.transformation
    """
    # set random seed
    np.random.seed(seed)
    for n in range(max_iter):
        # use different random seed for each iteration
        ants_seed = np.random.randint(1,10000)
        # initial rigid transformation
        trans_rigid = ants.registration(
            fixed=img_fix_ants,
            moving=img_move_ants,
            type_of_transform='QuickRigid', # 'AffineFast',
            aff_metric=aff_metric,
            outprefix=out_prefix+'_rigid_',
            random_seed=ants_seed,
            verbose=verbose)
        img_rigid_ants = ants.apply_transforms(
            fixed=img_fix_ants,
            moving=img_move_ants,
            transformlist=trans_rigid['fwdtransforms'],
            interpolator='linear')
        affine_rigid = ants_trans_to_mat(trans_rigid)
        
        # affine transformation
        trans_affine = ants.registration(
            fixed=img_fix_ants,
            moving=img_rigid_ants,
            type_of_transform='AffineFast', 
            aff_metric=aff_metric,
            outprefix=out_prefix+'_affine_',
            random_seed=ants_seed,
            verbose=verbose)
        img_align_ants = ants.apply_transforms(
            fixed=img_fix_ants,
            moving=img_rigid_ants,
            transformlist=trans_affine['fwdtransforms'],
            interpolator='linear')
        affine_mat = ants_trans_to_mat(trans_affine)

        # compute the dice overlap with atlas image
        mask_align = (img_align_ants.numpy()>0).astype(np.float32)
        mask_fix = (img_fix_ants.numpy()>0).astype(np.float32)
        align_dice = 2 * (mask_align * mask_fix).sum()\
            / (mask_align + mask_fix).sum()

        if align_dice >= min_dice:
            break
        
    # affine matrix after affine registration
    affine_mat = affine_rigid @ affine_mat @ affine_fix
    
    return img_align_ants, affine_mat, trans_rigid, trans_affine, align_dice

