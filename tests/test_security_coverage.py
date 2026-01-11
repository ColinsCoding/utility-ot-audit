from odat2.telecom.layout import Layout, Rect
from odat2.telecom.security.models import SensorSpec
from odat2.telecom.security.coverage import CoverageAnalyzer

def test_basic_coverage():
    layout=Layout(width=20,height=20,obstacles=[],cost_zones=[],turn_penalty=0.25)
    sensors=[SensorSpec("S1",10,10,6,360,0)]
    a=CoverageAnalyzer(layout,sensors)
    cov=a.coverage_grid()
    assert cov[10,14]>=1
