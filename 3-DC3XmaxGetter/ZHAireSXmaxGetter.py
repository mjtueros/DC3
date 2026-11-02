import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import utils

AiresInfo, AiresInp = utils.get_zhairestools_path()


def calculate_event_xmax(row, librarydir):
    """
    Calculate Xmax for one SQLite event.
    
    """
    ShowerName=row["ShowerName"]
    sryfile=librarydir+"/"+ShowerName+"/"+ShowerName+".sry"
    alt,distance,Xmaxx,Xmaxy,Xmaxz =AiresInfo.GetKmXmaxFromSry(sryfile)
    xmax=AiresInfo.GetSlantXmaxFromSry(sryfile) 
    return distance, alt, xmax