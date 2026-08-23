from fastapi import APIRouter, HTTPException

from backend.services.scanner import scan_directory, directory_to_dict
from backend.services.analysis import analyze_tree, analysis_to_dict


router = APIRouter()


@router.get("/scan")
def scan(path: str):
    try:
        result = scan_directory(path)
        return directory_to_dict(result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Directory does not exist.",
        )

    except NotADirectoryError:
        raise HTTPException(
            status_code=400,
            detail="Path is not a directory.",
        )

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied.",
        )
@router.get("/analyze")
def analyze(path: str):
    try:
        root = scan_directory(path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Directory does not exist.",
        )
    except NotADirectoryError:
        raise HTTPException(
            status_code=400,
            detail="Path is not a directory.",
        )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied.",
        )

    result = analyze_tree(root)

    return analysis_to_dict(result)