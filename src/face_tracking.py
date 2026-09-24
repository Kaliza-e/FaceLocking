"""Lock one enrolled identity using the existing Part 1 recognition pipeline."""
from dataclasses import dataclass
from enum import Enum, auto
from contextlib import closing

import cv2
import numpy as np

from src.align import align_face
from src.database import FaceDatabase
from src.config import LOGS
from src.event_log import TrackingLog
from src.face_signals import FaceSignalExtractor
from src.landmarks import draw_face
from src.workflow import camera, exiting, models, parser, read_frame, run


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    UNCERTAIN = auto()
    LOST = auto()


def center(box):
    """Center of an (x1, y1, x2, y2) box."""
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2.0


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1.0)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(self, target_name, detector, embedder, matcher,
                 verify_every=1, lost_timeout=24, ema_alpha=0.30,
                 dead_zone=0.07, threshold=0.45, margin=0.05, uncertain_grace=5):
        if target_name not in matcher.people:
            raise ValueError(f'Target {target_name!r} is not enrolled; use the exact enrolled name')
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError('Threshold must be [-1,1] and margin [0,2]')
        if (not isinstance(verify_every, int) or verify_every < 1
                or not isinstance(lost_timeout, int) or lost_timeout < 0):
            raise ValueError('verify_every must be a positive integer; lost_timeout must be nonnegative')
        if not 0 < ema_alpha <= 1 or not 0 <= dead_zone < 1:
            raise ValueError('ema_alpha must be (0,1] and dead_zone [0,1)')
        if not isinstance(uncertain_grace, int) or uncertain_grace < 0:
            raise ValueError('uncertain_grace must be a nonnegative integer')
        self.target_name = target_name
        self.detector, self.embedder, self.matcher = detector, embedder, matcher
        self.verify_every, self.lost_timeout = verify_every, lost_timeout
        self.ema_alpha, self.dead_zone = ema_alpha, dead_zone
        self.threshold, self.margin = threshold, margin
        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0
        self.uncertain_grace = uncertain_grace
        self.uncertain_frames = 0
        self.detection_confidence = None
        self.match_name = None
        self.match_score = None
        self.reason = 'Waiting for target'

    @staticmethod
    def box(face):
        # YuNet stores (x, y, width, height), unlike the PDF's detector.
        x, y, width, height = face.box
        return np.array([x, y, x + width, y + height], dtype=np.float32)

    def identity(self, frame, face):
        try:
            vector = self.embedder.embed(align_face(frame, face.points))
            return self.matcher.match(vector, self.threshold, self.margin)
        except ValueError:
            return 'Unknown', None

    def record_match(self, face, name, score):
        self.detection_confidence = float(face.confidence)
        self.match_name, self.match_score = name, score

    def acquire(self, frame, faces):
        best, best_score = None, -float('inf')
        for face in faces:
            name, score = self.identity(frame, face)
            if self.detection_confidence is None or (score is not None and
                    (self.match_score is None or score > self.match_score)):
                self.record_match(face, name, score)
            if name == self.target_name and score is not None and score > best_score:
                best, best_score = face, score
        if best is not None:
            self.record_match(best, self.target_name, best_score)
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        diagonal = max(float(np.linalg.norm(self.last_box[2:] - self.last_box[:2])), 1.0)
        ranked = []
        for face in faces:
            box = self.box(face)
            displacement = np.linalg.norm(center(box) - last_center) / diagonal
            ranked.append((iou(self.last_box, box) - 0.35 * displacement, face))
        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        self.detection_confidence = None
        uncertain = False
        faces = [face for face in self.detector.detect(frame) if min(face.box[2:]) >= 70]
        if not faces:
            self.match_name = self.match_score = None
            self.reason = 'No eligible face detected'
        else:
            self.reason = 'Target not verified'

        if self.state == LockState.SEARCHING:
            if not faces:
                self.match_name = self.match_score = None
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)
            if faces and candidate is None:
                self.detection_confidence = max(float(face.confidence) for face in faces)
                self.reason = 'Movement discontinuity'
            # Always verify after a gap or when multiple people could cross.
            verify = (self.state in (LockState.LOST, LockState.UNCERTAIN) or len(faces) > 1
                      or self.frame_index % self.verify_every == 0)
            if candidate is not None:
                self.detection_confidence = float(candidate.confidence)
            if candidate is not None and verify:
                name, score = self.identity(frame, candidate)
                self.record_match(candidate, name, score)
                if name != self.target_name:
                    # Grace only follows a currently observed, overlapping face.
                    # A gap, crossing, invalid embedding, or known distractor
                    # requires positive identity verification again.
                    uncertain = (name == 'Unknown' and score is not None
                                 and self.state in (LockState.LOCKED, LockState.UNCERTAIN)
                                 and len(faces) == 1
                                 and iou(self.last_box, self.box(candidate)) >= 0.5
                                 and self.uncertain_frames < self.uncertain_grace)
                    self.reason = ('Identity uncertain' if name == 'Unknown'
                                   else 'Different identity detected')
                    if not uncertain:
                        candidate = None
                else:
                    self.reason = 'Target verified'
            elif candidate is not None:
                self.reason = 'Following between identity checks'

        if candidate is None:
            if self.last_box is not None:
                self.lost_frames += 1
                self.state = LockState.LOST
                if self.lost_frames > self.lost_timeout:
                    self.state = LockState.SEARCHING
                    self.last_box = None
                    self.smooth_center = None
                    self.match_name = self.match_score = None
            return None, None

        if uncertain:
            self.uncertain_frames += 1
            self.state = LockState.UNCERTAIN
            self.reason = f'Identity uncertain ({self.uncertain_frames}/{self.uncertain_grace})'
        else:
            self.uncertain_frames = 0
            self.state = LockState.LOCKED
            if self.match_name == self.target_name:
                self.reason = 'Target verified'
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)
        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            self.smooth_center = (self.ema_alpha * raw_center
                                  + (1 - self.ema_alpha) * self.smooth_center)
        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape):
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2) / (width / 2))
        ey = float((self.smooth_center[1] - height / 2) / (height / 2))
        horizontal = 'LEFT' if ex < -self.dead_zone else 'RIGHT' if ex > self.dead_zone else 'CENTER'
        vertical = 'UP' if ey < -self.dead_zone else 'DOWN' if ey > self.dead_zone else 'CENTER'
        return TrackingSignal(ex, ey, horizontal, vertical)


def status_panel(frame, tracker, position, signals, blink_total, settings):
    """Keep live readings and calibration settings visible beside the camera."""
    lines = [f'{tracker.state.name}: {tracker.target_name}',
             tracker.reason,
             (f'Detection confidence: {tracker.detection_confidence:.3f}'
              if tracker.detection_confidence is not None else 'Detection confidence: N/A'),
             (f'Best match: {tracker.match_name} | {tracker.match_score:.3f}'
              if tracker.match_score is not None else 'Best match: N/A'),
             f'{position.horizontal} / {position.vertical}' if position else 'Position: N/A',
             f'error=({position.error_x:+.2f}, {position.error_y:+.2f})' if position else 'error: N/A',
             '',
             ('SMILE' if signals.smiling else 'NEUTRAL') if signals else 'Smile: N/A',
             (('EYES CLOSED' if signals.eyes_closed else 'EYES CLOSING')
              if signals.ear < settings.ear_threshold else 'EYES OPEN') if signals else 'Eyes: N/A',
             f'BLINKS: {blink_total}',
             f'EAR: {signals.ear:.3f}' if signals else 'EAR: N/A',
             f'Smile score: {signals.smile_score:.3f}' if signals else 'Smile score: N/A',
             '', 'CALIBRATION',
             f'EAR threshold: {settings.ear_threshold:g}',
             f'Blink frames: {settings.blink_min_frames}-{settings.blink_max_frames}',
             f'Closed frames: {settings.closed_frames}',
             f'Smile on/off: {settings.smile_on:g} / {settings.smile_off:g}',
             f'Match threshold/margin: {tracker.threshold:g} / {tracker.margin:g}',
             f'Verify every: {tracker.verify_every} frame(s)',
             f'Lost timeout: {tracker.lost_timeout} frames',
             f'Uncertain grace: {tracker.uncertain_grace} frames',
             f'EMA: {tracker.ema_alpha:g} | Dead zone: {tracker.dead_zone:g}',
             '', 'Q / Escape: quit']
    panel_width = 390
    view = np.full((max(frame.shape[0], len(lines) * 27 + 20),
                    frame.shape[1] + panel_width, 3), 24, dtype=np.uint8)
    view[:frame.shape[0], :frame.shape[1]] = frame
    for index, text in enumerate(lines):
        color = (0, 220, 160) if index == 0 else (230, 230, 230)
        cv2.putText(view, text, (frame.shape[1] + 12, 27 * (index + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return view


def main():
    cli = parser(__doc__)
    cli.add_argument('--target', required=True, help='Exact enrolled name to lock')
    cli.add_argument('--log-dir', default=str(LOGS), help='Directory for per-session CSV logs')
    cli.add_argument('--threshold', type=float, default=0.45, help='Part 1 cosine similarity threshold')
    cli.add_argument('--margin', type=float, default=0.05)
    cli.add_argument('--verify-every', type=int, default=3,
                     help='Identity check interval; larger values can briefly follow an unverified face')
    cli.add_argument('--lost-timeout', type=int, default=24, help='Missed frames allowed before searching again')
    cli.add_argument('--uncertain-grace', type=int, default=5,
                     help='Consecutive uncertain identity frames allowed with continuous overlap; 0 disables')
    cli.add_argument('--ema-alpha', type=float, default=0.30)
    cli.add_argument('--dead-zone', type=float, default=0.07)
    cli.add_argument('--ear-threshold', type=float, default=0.21)
    cli.add_argument('--blink-min-frames', type=int, default=2)
    cli.add_argument('--blink-max-frames', type=int, default=7)
    cli.add_argument('--closed-frames', type=int, default=8)
    cli.add_argument('--smile-on', type=float, default=0.38)
    cli.add_argument('--smile-off', type=float, default=0.35)
    args = cli.parse_args()
    with TrackingLog(args.log_dir, vars(args)) as log:
        print(f'Session log: {log.path}')
        tracking_session(args, cli, log)


def tracking_session(args, cli, log):
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if not db.people:
        cli.error('No faces enrolled. Run python -m src.enroll --name "Your Name" first.')
    tracker = LockedFaceTracker(args.target, detector, embedder, db,
                                args.verify_every, args.lost_timeout, args.ema_alpha,
                                args.dead_zone, args.threshold, args.margin, args.uncertain_grace)
    window = 'Face identity lock'
    print(f'Tracking {args.target!r}. Q or Escape quits. Scores use Part 1 cosine matching.')
    blink_total = 0
    with closing(FaceSignalExtractor(
            ear_threshold=args.ear_threshold, blink_min_frames=args.blink_min_frames,
            blink_max_frames=args.blink_max_frames, closed_frames=args.closed_frames,
            smile_on=args.smile_on, smile_off=args.smile_off)) as extractor, camera(args.camera) as cap:
        while True:
            frame = read_frame(cap)
            face, position = tracker.update(frame)
            face_state = None
            if face is not None:
                face_state = extractor.analyze(frame, tracker.box(face))
                if face_state is not None and face_state.blink:
                    blink_total += 1
            else:
                extractor.reset()
            log.frame(tracker, position, face_state, blink_total, args.ear_threshold)
            # All inference uses the original pixels before overlays are drawn.
            color = (0, 200, 0) if tracker.state == LockState.LOCKED else (0, 140, 255)
            if face is not None:
                draw_face(frame, face, f'{args.target} | {tracker.state.name}', color)
                cx, cy = np.rint(tracker.smooth_center).astype(int)
                cv2.circle(frame, (cx, cy), 5, (255, 170, 0), -1)
            height, width = frame.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(frame, (int(width * (0.5 - dz / 2)), int(height * (0.5 - dz / 2))),
                          (int(width * (0.5 + dz / 2)), int(height * (0.5 + dz / 2))),
                          (120, 120, 120), 1)
            cv2.imshow(window, status_panel(frame, tracker, position, face_state, blink_total, args))
            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == '__main__':
    run(main)
