package io.bioimage.modelrunner.transformations.sam;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

public class NMS {
	public static List<Integer> greedyNMS(List<double[]> boxes, List<Double> scores, double iou_threshold) {
	    List<Integer> indices = new ArrayList<>();
	    PriorityQueue<Integer> pq = new PriorityQueue<>((i, j) -> -Double.compare(scores.get(i), scores.get(j)));
	    int n = scores.size();

	    // Add all boxes to the priority queue
	    for (int i = 0; i < n; i++) {
	        pq.add(i);
	    }

	    while (!pq.isEmpty()) {
	        int index = pq.poll();
	        indices.add(index);

	        // Remove boxes with high IoU
	        Iterator<Integer> iter = pq.iterator();
	        while (iter.hasNext()) {
	            int j = iter.next();
	            double iou = compute_iou(boxes.get(index), boxes.get(j));
	            if (iou > iou_threshold) {
	                iter.remove();
	            }
	        }
	    }

	    return indices;
	}

	public static double compute_iou(double[] box1, double[] box2) {
	    // Calculate intersection box
	    double x1 = Math.max(box1[0], box2[0]);
	    double y1 = Math.max(box1[1], box2[1]);
	    double x2 = Math.min(box1[2], box2[2]);
	    double y2 = Math.min(box1[3], box2[3]);
	    double intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

	    // Calculate union box
	    double area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
	    double area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
	    double union = area1 + area2 - intersection;

	    // Calculate IoU
	    double iou = intersection / union;
	    return iou;
	}
}
