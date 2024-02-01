import os
import glob
import numpy as np

def create_submission(seismic_filenames: list, prediction: list, submission_path: str):
    """Function to create submission file out of all test predictions in one list

    Parameters:
        seismic_filenames: list of survey .npy filenames used for perdiction
        prediction: list with 3D np.ndarrays of predicted missing parts
        submission_path: path to save submission

    Returns:
        None
    """
    
    submission_keys = ['target-92515355.npy-i_0', 'target-92515355.npy-i_1', 'target-92515355.npy-i_2',
            'target-92515355.npy-x_0', 'target-92515355.npy-x_1', 'target-92515355.npy-x_2',
            'target-92515931.npy-i_0', 'target-92515931.npy-i_1', 'target-92515931.npy-i_2',
            'target-92515931.npy-x_0', 'target-92515931.npy-x_1', 'target-92515931.npy-x_2',
            'target-92551495.npy-i_0', 'target-92551495.npy-i_1', 'target-92551495.npy-i_2',
            'target-92551495.npy-x_0', 'target-92551495.npy-x_1', 'target-92551495.npy-x_2',
            'target-92515357.npy-i_0', 'target-92515357.npy-i_1', 'target-92515357.npy-i_2',
            'target-92515357.npy-x_0', 'target-92515357.npy-x_1', 'target-92515357.npy-x_2',
            'target-92508315.npy-i_0', 'target-92508315.npy-i_1', 'target-92508315.npy-i_2',
            'target-92508315.npy-x_0', 'target-92508315.npy-x_1', 'target-92508315.npy-x_2',
            'target-92515353.npy-i_0', 'target-92515353.npy-i_1', 'target-92515353.npy-i_2',
            'target-92515353.npy-x_0', 'target-92515353.npy-x_1', 'target-92515353.npy-x_2',
            'target-92515932.npy-i_0', 'target-92515932.npy-i_1', 'target-92515932.npy-i_2',
            'target-92515932.npy-x_0', 'target-92515932.npy-x_1', 'target-92515932.npy-x_2',
            'target-92568236.npy-i_0', 'target-92568236.npy-i_1', 'target-92568236.npy-i_2',
            'target-92568236.npy-x_0', 'target-92568236.npy-x_1', 'target-92568236.npy-x_2',
            'target-92551498.npy-i_0', 'target-92551498.npy-i_1', 'target-92551498.npy-i_2',
            'target-92551498.npy-x_0', 'target-92551498.npy-x_1', 'target-92551498.npy-x_2',
            'target-92568004.npy-i_0', 'target-92568004.npy-i_1', 'target-92568004.npy-i_2',
            'target-92568004.npy-x_0', 'target-92568004.npy-x_1', 'target-92568004.npy-x_2',
            'target-92551496.npy-i_0', 'target-92551496.npy-i_1', 'target-92551496.npy-i_2',
            'target-92551496.npy-x_0', 'target-92551496.npy-x_1', 'target-92551496.npy-x_2',
            'target-92568235.npy-i_0', 'target-92568235.npy-i_1', 'target-92568235.npy-i_2',
            'target-92568235.npy-x_0', 'target-92568235.npy-x_1', 'target-92568235.npy-x_2',
            'target-92515930.npy-i_0', 'target-92515930.npy-i_1', 'target-92515930.npy-i_2',
            'target-92515930.npy-x_0', 'target-92515930.npy-x_1', 'target-92515930.npy-x_2',
            'target-92515356.npy-i_0', 'target-92515356.npy-i_1', 'target-92515356.npy-i_2',
            'target-92515356.npy-x_0', 'target-92515356.npy-x_1', 'target-92515356.npy-x_2',
            'target-92551497.npy-i_0', 'target-92551497.npy-i_1', 'target-92551497.npy-i_2',
            'target-92551497.npy-x_0', 'target-92551497.npy-x_1', 'target-92551497.npy-x_2']

    submission = {key: None for key in submission_keys}
    for sample_name, sample_prediction in zip(seismic_filenames, prediction):
        i_slices_index = (np.array([.25, .5, .75]) * sample_prediction.shape[0]).astype(int)
        i_slices_names = [f'target-{sample_name}-i_{n}' for n in range(0,3)]
        i_slices = [sample_prediction[s, :, :].astype(np.uint8) for s in i_slices_index]
        submission.update(dict(zip(i_slices_names, i_slices)))

        x_slices_index = (np.array([.25, .5, .75]) * sample_prediction.shape[1]).astype(int)
        x_slices_names = [f'target-{sample_name}-x_{n}' for n in range(0,3)]
        x_slices = [sample_prediction[:, s, :].astype(np.uint8) for s in x_slices_index]
        submission.update(dict(zip(x_slices_names, x_slices)))
    
    
    np.savez(submission_path, **submission)
    
if __name__=="__main__":
    np_output_path = "/home/aittgp/vutt/workspace/patch_planet/Image-Colorizer-Pix2Pix/np_output_fixed"
    np_outputs = glob.glob(os.path.join(np_output_path, "*.npy"))
    
    submission_path = "vutt_submission.npz"
    
    predictions = []
    seismic_filenames = []
    
    for file_path in np_outputs:
        
        file_name = file_path.split("/")[-1].split("-")[1]
        print(file_name)
        seismic_filenames.append(file_name)
        
        prediction = np.load(file_path)
        print(prediction.shape)
        predictions.append(prediction)
        
    print(len(predictions))
    print(len(seismic_filenames))
    
    create_submission(seismic_filenames, predictions, submission_path)
        
    
    