import sys, os
import shutil   # 고수준 파일 연산.
import glob     # 글로브(glob) 패턴은 와일드카드 문자로 여러 파일 이름의 집합을 지정

def file_name_only(path_str):
    '''
    full_path에서 directory를 제거한 파일명.
    '''
    return os.path.basename(path_str)


def file_name_except_extention(file_name):
    '''
    파일명에서 확장자를 제거한 파일이름.
    '''
    return os.path.splitext(file_name)[0]


def file_extension_only(file_name):
    '''
    파일명에서 확장자를 구함.
    '''    
    return os.path.splitext(file_name)[1][1:]


def concat_file_name(file_path, file_ext):
    return f"{file_path}.{file_ext}"


def extension_change(file_path, file_ext):
    file_name = file_name_except_extention(file_path)
    return f"{file_name}.{file_ext}"


def dir_path_change(from_path, to_path, to_ext, check_duplicate=True):
    '''
    from_path의 파일을 to_path의 ext를 사용하여 저장.
    '''
    file_name = file_name_only(from_path)
    file_name = file_name_except_extention(file_name)

    file_name = os.path.join(to_path, file_name)
    file_name = concat_file_name(file_name, to_ext)

    new_path = not_duplicated_path(file_name)

    return new_path


def not_duplicated_path(save_path):
    file_name = file_name_except_extention(save_path)
    file_ext = file_extension_only(save_path)

    new_path = concat_file_name(file_name, file_ext)
    idx = 0
    while os.path.exists(new_path):
        new_path = file_name + f"_{idx:03d}." + file_ext

        if os.path.exists(new_path):
            idx = idx + 1
            continue
        else:
            break
        
    return new_path

