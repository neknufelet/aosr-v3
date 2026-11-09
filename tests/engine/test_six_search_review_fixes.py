"""B2a 審查第 1 輪：混合拒收、產品行程不可估與 CLI 換版邊界。"""
from __future__ import annotations

from collections.abc import Callable
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tests.engine._scheme_cache import shared_json
from tests.engine._six_search_cli_case import model as model, operators as operators
from tests.engine.test_material_gradient_cli import _env
from tests.engine.test_material_gradient_endpoint_cli import _inputs, _surrogate


def _probe(directory: Path, mode: str) -> dict[str,object]:
    done = subprocess.run([sys.executable,'-m','tests.engine._six_search_review_probe',mode,str(directory)],
        cwd=directory,env=_env(directory),capture_output=True,text=True,timeout=180)
    assert done.returncode==0,done.stdout+done.stderr
    return cast(dict[str,object],json.loads((directory/(mode+'.json')).read_text()))


@pytest.fixture(scope='module')
def measured(request: pytest.FixtureRequest,tmp_path_factory: pytest.TempPathFactory,worker_id: str
             ) -> Callable[[str],dict[str,object]]:
    def get(mode: str) -> dict[str,object]:
        def produce() -> str:
            directory=tmp_path_factory.mktemp('six-review-'+mode)
            if mode=='cli':
                _inputs(directory)
                _surrogate(directory,request)
            return json.dumps(_probe(directory,mode))
        return cast(dict[str,object],json.loads(shared_json(tmp_path_factory,worker_id,'six-review-'+mode,produce)))
    return get












@pytest.mark.parametrize(('version','expected'),(
    ('aosr.material_gradient_ledger.v6','aosr.material_gradient_templates.v3'),
    ('aosr.material_gradient_ledger.v5','aosr.material_gradient_templates.v2'),
    ('aosr.material_gradient_ledger.v4','aosr.material_gradient_templates.v1')))
def test_template_header_uses_explicit_ledger_version_mapping(version: str,expected: str) -> None:
    from aosr.gradient import run
    from aosr.geometry.shoebox import Wall
    from tests.engine._material_gradient_loop_probe import synthetic_header
    from tests.engine._material_gradient_templates_probe import kept_walls
    from tests.engine.test_material_gradient import DESCENT_CASE,_designs,_settings
    from tests.engine.test_material_gradient_templates import KEPT
    header=synthetic_header(_designs(DESCENT_CASE),_settings())
    constants=header.score_constants
    assert constants is not None
    low=constants.model_copy(update={'category_weights':None})
    header=header.model_copy(update={'ledger_version':version,
        'score_constants':constants if version.endswith('v6') else low if version.endswith('v5') else None})
    materials=kept_walls(KEPT)
    kept={wall:materials[wall.wall_name()] for wall in Wall.all()}
    result=run.template_header_for(header,_settings(),kept,base_ledger_sha256='e'*64)
    assert result.ledger_version==expected
