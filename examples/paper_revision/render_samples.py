"""Render the retained five- and seven-loss sample comparisons and expansion figure."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from examples.paper_revision.illustrations.layouts import rings,sine,expansion

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--part',choices=['toys','expansion','all'],default='all');a=p.parse_args()
    if a.part in ['toys','all']:
        methods=['MSE','ED','SD','CSD','GSD'];seven=methods+['MMD','SW1']
        rings(a.work/'toys',a.out,methods,'fig08_generative_loss.png');rings(a.work/'toys',a.out,seven,'figS09_toy_circles.png')
        sine(a.work/'toys',a.out,methods,'fig09c_sinusoidal_full.png');sine(a.work/'toys',a.out,seven,'figS10_toy_sine.png')
    if a.part in ['expansion','all']:expansion(a.work,a.out)
