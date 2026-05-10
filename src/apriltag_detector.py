import cv2
from pupil_apriltags import Detector


DEFAULT_TAG_IDS = (0, 1, 2, 3)


class AprilTagPoseEstimator:
    def __init__(self, tag_ids=DEFAULT_TAG_IDS):
        self.detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=1,
        )
        self.tag_ids = tuple(tag_ids)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detection_by_id = {
            int(det.tag_id): det
            for det in detections
            if int(det.tag_id) in self.tag_ids
        }

        annotated_frame = frame.copy()
        for det in detection_by_id.values():
            self._draw_detection(annotated_frame, det)

        missing_ids = [tag_id for tag_id in self.tag_ids if tag_id not in detection_by_id]
        return len(missing_ids) == 0, annotated_frame, detection_by_id, missing_ids

    @staticmethod
    def _draw_detection(frame, det):
        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.putText(
            frame,
            f"ID:{det.tag_id}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
