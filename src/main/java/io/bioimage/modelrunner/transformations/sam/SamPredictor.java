package io.bioimage.modelrunner.transformations.sam;

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.transformations.ZeroMeanUnitVarianceTransformation;
import net.imglib2.img.Img;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class SamPredictor {
	
	private static final double eps = 1e-10;

    private long[] inputSize;
    private long[] originalSize;
	private boolean isImageSet = false;
    private int origH = 0;
    private int origW = 0;
    private int inputH = 0;
    private int inputW = 0;
    
    public SamPredictor() {
    	
    }
	
	
	public < R extends RealType< R > & NativeType< R > > void setTorchImage(Img<R> image) throws LoadEngineException, Exception {
		resetImage();
		inputSize = new long[] {image.dimensionsAsLongArray()[2], image.dimensionsAsLongArray()[2]}; 
		preprocess((Img<FloatType>) image);
		
		// TODO get encoder and run model
		EngineInfo engineInfo = EngineInfo.defineDLEngine("pytorch", "1.13.1", true, true);
		Model model = Model.createDeepLearningModel("path/to/cache", "path/to/cache/model.pt", engineInfo);
		model.loadModel();
		model.runModel(null, null);
		isImageSet = true;
		
	}
	
	public static void preprocess(Img<FloatType> image) {
		final float[] meanStd = ZeroMeanUnitVarianceTransformation.meanStd( image );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];
		LoopBuilder.setImages( image ).multiThreaded()
			.forEachPixel( i -> i.set( ( i.get() - mean ) / ( std + (float) eps ) ) );
		// TODO add padding
		// TODO add padding
		// TODO add padding
		// TODO add padding
		// TODO add padding
	}
	
	private void resetImage() {
	    origH = 0;
	    origW = 0;
	    inputH = 0;
	    inputW = 0;
	}
}
