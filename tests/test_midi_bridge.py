import importlib.util
from pathlib import Path
import tempfile
import unittest
import mido

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('midi_bridge',ROOT/'tools/midi_bridge.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


class MidiBridge(unittest.TestCase):
    def test_crop_preserves_tempo_beats_roles_and_closes_overlapping_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            source,out=Path(temp)/'source.mid',Path(temp)/'crop.mid';m=mido.MidiFile(type=1,ticks_per_beat=960)
            conductor=mido.MidiTrack();m.tracks.append(conductor)
            conductor.extend([mido.MetaMessage('set_tempo',tempo=500000),mido.MetaMessage('set_tempo',tempo=1000000,time=960)])
            lane=mido.MidiTrack();m.tracks.append(lane)
            lane.extend([mido.MetaMessage('track_name',name='Instrument Melody'),mido.Message('note_on',note=60,velocity=100,time=240),
                mido.Message('note_on',note=60,velocity=100,time=240),mido.Message('note_off',note=60,time=1200),mido.Message('note_off',note=60,time=240)])
            m.save(source);report=bridge.crop_midi(source,out,6);cropped,notes=bridge.midi_notes(out)
            self.assertEqual([(n['onset_step'],n['duration_steps']) for n in notes],[(1,5),(2,4)])
            self.assertEqual(cropped.tracks[1].name,'Instrument Melody')
            self.assertEqual([(x.tempo,x.time) for x in cropped.tracks[0] if x.type=='set_tempo'],[(500000,0),(1000000,960)])
            self.assertAlmostEqual(report['duration_seconds'],1.0)

    def test_event_audit_detects_timing_mismatch_and_role(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'melody.mid';m=mido.MidiFile(ticks_per_beat=480);t=mido.MidiTrack();m.tracks.append(t)
            t.extend([mido.Message('note_on',note=64,velocity=100,time=240),mido.Message('note_off',note=64,time=480)]);m.save(path)
            events=[dict(subbeat=2,notes=[dict(pitch=64,track=1,duration_steps=4)])]
            self.assertTrue(bridge.audit_events(path,events,'instrumental')['beat_positions_pitches_durations_match'])
            self.assertFalse(bridge.audit_events(path,events,'vocal')['beat_positions_pitches_durations_match'])
            events[0]['subbeat']=3
            self.assertFalse(bridge.audit_events(path,events,'instrumental')['beat_positions_pitches_durations_match'])

    def test_refuses_unterminated_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'broken.mid';m=mido.MidiFile();t=mido.MidiTrack();m.tracks.append(t);t.append(mido.Message('note_on',note=60,velocity=100));m.save(path)
            with self.assertRaises(ValueError):bridge.midi_notes(path)


if __name__=='__main__':unittest.main()
