from contextlib import closing
import json

import numpy as np
import pytest

from test_afv_benchmark import tool
from winding_field import afv
from winding_field.artifacts import Tiles


def test_support_cache_survives_adjacent_source_blocks(tmp_path,monkeypatch):
    config=tool().generated_input(tmp_path/'run',1000,1);loads=[];original=Tiles.load
    def load(self,key,names=None):
        if names==('support',):loads.append(key)
        return original(self,key,names)
    monkeypatch.setattr(Tiles,'load',load)
    result=afv.import_fibres(tmp_path/'run',config)
    assert result['count']==2000
    assert len(loads)<=3


def test_support_cache_eviction_and_ownership(tmp_path):
    from winding_field.afv_clip import SupportTiles
    tool().generated_input(tmp_path/'run',1,1)
    with closing(Tiles(tmp_path/'run/maps')) as maps:
        cache=SupportTiles(maps,cache_bytes=512)
        first=cache.load((0,0,0))['support']
        assert not first.flags.writeable
        cache.load((0,0,1));assert cache.bytes<=512
        np.testing.assert_array_equal(cache.load((0,0,0))['support'],first)
        cache.close()
        assert maps.load((0,0,0))['support'].all()
        tiny=SupportTiles(maps,cache_bytes=1)
        tiny.load((0,0,0));assert tiny.bytes==0


def test_cached_support_cannot_hide_mid_import_map_mutation(tmp_path,monkeypatch):
    config=tool().generated_input(tmp_path/'run',4,1)
    original=afv._pieces;changed=False
    def pieces(*args,**kwargs):
        nonlocal changed
        yield from original(*args,**kwargs)
        if not changed:
            changed=True
            with closing(Tiles(tmp_path/'run/maps')) as maps:
                data=maps.load((0,0,0));data['support'][0,0,0]=False
                maps.write((0,0,0),data)
    monkeypatch.setattr(afv,'_pieces',pieces)
    with pytest.raises(ValueError,match='maps changed'):
        afv.import_fibres(tmp_path/'run',config)
    assert not (tmp_path/'run/FIBRES.json').exists()
