"""Helper functions for file processing."""
from os import fdopen, remove
from shutil import move
from tempfile import mkstemp


def replace(file_path, pattern, subst):
    """
    Replace strings in a file.
    
    Parametrs:
        pattern (str, iterable): pattern to replace.
        subst (str, iterable): subtitution.
    
    """
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


def save_dice(out_dice_path, img_names, dice_values):
    """
    Save DICE values into a csv file.
    Tissue type should be in this order: csf,gm,wm.
    
    Parameters:
        out_dice_path (str): output file path.
        img_names (list[str]): list of volumes used for dice calculation.
        dice_values (numpy.ndarray): array with dice coefficeints for each tissue type.
    
    Returns: None.
    """
    with open(out_dice_path, 'w+') as out_f:
        out_f.write('img,csf,gm,wm,\n')
        for index, row in enumerate(dice_values): 
            out_f.write(img_names[index] + ',' + ','.join(str(j) for j in row) + ',\n')
