package io.bioimage.modelrunner.transformations.sam;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class MaskData {
	private HashMap<String, Object> stats; 
	private static String ILLEGAL_ARG_MESSAGE = "MaskData only "
				+ "supports list, ImgLib2 RandomAccessibleIntervals "
				+ "or Tensors.";
	
	public MaskData(HashMap<String, Object> kwargs) {
		checkArgs(kwargs);
		stats = kwargs;
	}
	
	public MaskData() {
		stats = new HashMap<String, Object>();
	}

	private void checkArgs(HashMap<String, Object> kwargs) {
		for (Entry<String, Object> entry : kwargs.entrySet()) {
			if (!checkArg(entry.getValue()))
				throw new IllegalArgumentException(ILLEGAL_ARG_MESSAGE + " Input '" + entry.getKey() 
						+ "' is not of those types.");
		}
	}
	
	private boolean checkArg(Object arg) {
		boolean isList = (arg instanceof ArrayList);
		boolean isImg = (arg instanceof RandomAccessibleInterval);
		boolean isTensor = (arg instanceof Tensor);
		if (!isList && !isImg && !isTensor)
			return false;
		return true;
			
	}
	
	public void set(String key, Object item) {
		if (!checkArg(item))
			throw new IllegalArgumentException(ILLEGAL_ARG_MESSAGE);
		stats.put(key, item);
	}
	
	public void del(String key) {
		if (this.stats.get(key) != null)
			this.stats.remove(key);
	}
	
	public Object get(String key) {
		return this.stats.get(key);
	}
	
	public void filter(boolean[] keep) {
		for (Entry<String, Object> entry : this.stats.entrySet()) {
			if (entry.getValue() == null) {
				return;
			} else if (entry.getValue() instanceof RandomAccessibleInterval) {
				stats.put(entry.getKey(), keepPlanes((RandomAccessibleInterval<FloatType>) entry.getValue(), keep));
			} else if (entry.getValue() instanceof Tensor) {
				RandomAccessibleInterval rai = ((Tensor) entry.getValue()).getData();
				Tensor nTensor = Tensor.build(((Tensor) entry.getValue()).getName(), ((Tensor) entry.getValue()).getAxesOrderString(), 
						keepPlanes(rai, keep));
				((Tensor) entry.getValue()).close();
				stats.put(entry.getKey(), nTensor);
			} else if (entry.getValue() instanceof List) {
				List<Object> list = (List<Object>) entry.getValue();
				List<Object> nList = new ArrayList<Object>();
				for (int i = 0; i < list.size(); i ++) {
					if (keep[i])
						nList.add(list.get(i));
				}
			}
		}
	}
	
	public void cat(MaskData data) {
		for (Entry<String, Object> entry : data.stats.entrySet()) {
			if (stats.get(entry.getKey()) == null || !stats.keySet().contains(entry.getKey())) {
				stats.put(entry.getKey(), entry.getValue());
			} else if (stats.get(entry.getKey()) instanceof RandomAccessibleInterval) {
				RandomAccessibleInterval<FloatType> res = concat((RandomAccessibleInterval<FloatType>) stats.get(entry.getKey()), (RandomAccessibleInterval<FloatType>) entry.getValue());
				stats.put(entry.getKey(), res);
			} else if (stats.get(entry.getKey()) instanceof Tensor) {
				RandomAccessibleInterval rai1 = ((Tensor) stats.get(entry.getKey())).getData();
				RandomAccessibleInterval rai2 = ((Tensor) entry.getValue()).getData();
				Tensor nTensor = Tensor.build(((Tensor) entry.getValue()).getName(), ((Tensor) entry.getValue()).getAxesOrderString(), 
						concat(rai1, rai2));
				((Tensor) entry.getValue()).close();
				stats.put(entry.getKey(), nTensor);
			} else if (stats.get(entry.getKey()) instanceof List) {
				List<Object> list = (List<Object>) stats.get(entry.getKey());
				for (Object obj : ((List<Object>) entry.getValue())) {
					list.add(obj);
				}
				stats.put(entry.getKey(), list);
			}
		}
	}
	
	private static RandomAccessibleInterval<FloatType> keepPlanes(RandomAccessibleInterval<FloatType> rai, boolean[] keep) {
		long[] orShape = rai.dimensionsAsLongArray();
		List<Long> keptSlices = new ArrayList<Long>();
		long nSlices = 0;
		for (long i = 0; i < keep.length; i ++) {
			if (keep[(int) i])
				keptSlices.add(i);
		}
		orShape[0] = keptSlices.size();
		Img<FloatType> nRai = 
				new ArrayImgFactory<>(new FloatType()).create(orShape);
		Cursor<FloatType> cursor = nRai.cursor();
		while (cursor.hasNext()) {
			long[] oldPos = cursor.positionAsLongArray();
			oldPos[0] = keptSlices.get((int) oldPos[0]);
			cursor.get().set(rai.getAt(oldPos));
		}
		return nRai;
	}
	
	private static RandomAccessibleInterval<FloatType> concat(RandomAccessibleInterval<FloatType> rai1,
			RandomAccessibleInterval<FloatType> rai2){
		long[] shape1 = rai1.dimensionsAsLongArray();
		long[] shape2 = rai2.dimensionsAsLongArray();
		if (shape1.length != shape2.length)
			throw new IllegalArgumentException("The two images need to have the same number of dims");
		long[] fShape = shape1;
		fShape[0] = shape1[0] + shape2[0];
		Img<FloatType> nImg = new ArrayImgFactory<>(new FloatType()).create(fShape);
		Cursor<FloatType> cursor = ((IterableInterval<FloatType>) rai1).cursor();
		while (cursor.hasNext()) {
			long[] pos = cursor.positionAsLongArray();
			nImg.getAt(pos).set(cursor.get());
		}
		cursor = ((IterableInterval<FloatType>) rai2).cursor();
		while (cursor.hasNext()) {
			long[] pos = cursor.positionAsLongArray();
			pos[0] = pos[0] + shape1[0];
			nImg.getAt(pos).set(cursor.get());
		}
		return nImg;
	}

}
