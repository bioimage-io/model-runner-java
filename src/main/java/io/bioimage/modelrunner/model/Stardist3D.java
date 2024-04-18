/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.model;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.view.Views;

public class Stardist3D {
	
	ModelDescriptor descriptor;
	
	private final int channels;
	
	private final float nms_threshold;
	
	private final float prob_threshold;
	
	private static final List<String> STARDIST_DEPS = Arrays.asList(new String[] {"python=3.10", "stardist", "numpy", "appose"});
	
	private static final List<String> STARDIST_CHANNELS = Arrays.asList(new String[] {"conda-forge", "default"});
	
	private static final String STARDIST3D_PATH_IN_RESOURCES = "ops/stardist_postprocessing/";
	
	private static final String STARDIST3D_SCRIPT_NAME= "stardist_postprocessing_3D.py";
	
	private static final String STARDIST3D_METHOD_NAME= "stardist_postprocessing";
	
	public Stardist3D() {
		this.channels = 1;
		// TODO get from config??
		this.nms_threshold = 0;
		this.prob_threshold = 0;
	}
	
	private Stardist3D(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get("stardist");
    	Map<String, Object> stardistConfig = (Map<String, Object>) stardistMap.get("config");
    	Map<String, Object> stardistThres = (Map<String, Object>) stardistMap.get("thresholds");
		this.channels = (int) stardistConfig.get("n_channel_in");;
		this.nms_threshold = new Double((double) stardistThres.get("nms")).floatValue();
		this.prob_threshold = new Double((double) stardistThres.get("prob")).floatValue();
	}
	
	public static Stardist3D fromBioimageioModel(String modelPath) throws ModelSpecsException {
		ModelDescriptor descriptor = ModelDescriptor.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME, false);
		return new Stardist3D(descriptor);
	}
	
	public static Stardist3D fromPretained(String pretrainedModel, boolean forceInstall) throws IOException, InterruptedException, ModelSpecsException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceInstall);
	}
	
	/**
	 * @param pretrainedModel
	 * @param installDir
	 * @param forceInstall
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ModelSpecsException
	 */
	public static Stardist3D fromPretained(String pretrainedModel, String installDir, boolean forceInstall) throws IOException, 
																					InterruptedException, 
																					ModelSpecsException {
		if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet") && !forceInstall) {
			ModelDescriptor md = ModelDescriptor.getModelsAtLocalRepo().stream()
					.filter(mm ->mm.getName().equals(pretrainedModel)).findFirst().orElse(null);
			if (md != null) return new Stardist3D(md);
			String path = BioimageioRepo.connect().downloadByName("StarDist Plant Nuclei 3D ResNet", installDir);
			return Stardist3D.fromBioimageioModel(path);
		} else if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet")) {
			String path = BioimageioRepo.connect().downloadByName("StarDist Plant Nuclei 3D ResNet", installDir);
			return Stardist3D.fromBioimageioModel(path);
		} else {
			throw new IllegalArgumentException("There is no Stardist3D model called: " + pretrainedModel);
		}
	}
	
	private <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image) {
		if (image.dimensionsAsLongArray().length == 3 && this.channels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYCZ");
		else if (image.dimensionsAsLongArray().length != 4 && this.channels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYCZ");
		else if (image.dimensionsAsLongArray().length == 4 && image.dimensionsAsLongArray()[2] != channels)
			throw new IllegalArgumentException("This Stardist3D model requires " + channels + " channels.");
		else if (image.dimensionsAsLongArray().length > 4 || image.dimensionsAsLongArray().length < 2)
			throw new IllegalArgumentException("Stardist3D model requires an image with dimensions XYCZ.");
	}
	
	public <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> predict(RandomAccessibleInterval<T> image) throws ModelSpecsException, LoadModelException,
																				LoadEngineException, IOException, 
																				RunModelException, InterruptedException {
		checkInput(image);
		if (image.dimensionsAsLongArray().length == 3) {
			image = Views.addDimension(image, 0, 0);
			image = Utils.transpose(image);
			image = Views.addDimension(image, 0, 0);
		} else if (image.dimensionsAsLongArray().length == 4) {
			image = Views.permute(image, 1, 2);
			image = Views.addDimension(image, 0, 1);
			image = Views.addDimension(image, 0, 0);
			image = Utils.transpose(image);
		}

		Tensor<T> inputTensor = Tensor.build("input", "bzyxc", image);
		Tensor<T> outputTensor = Tensor.buildEmptyTensor("output", "bzyxc");

		List<Tensor<?>> inputList = new ArrayList<Tensor<?>>();
		List<Tensor<?>> outputList = new ArrayList<Tensor<?>>();
		inputList.add(inputTensor);
		outputList.add(outputTensor);
		
		Model model = Model.createBioimageioModel(this.descriptor.getModelPath());
		model.loadModel();
		model.runModel(inputList, outputList);

		return Utils.transpose(Cast.unchecked(postProcessing(outputList.get(0).getData())));
	}
	
	public <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> postProcessing(RandomAccessibleInterval<T> image) throws IOException, InterruptedException {
		Mamba mamba = new Mamba();
		String envPath = mamba.getEnvsDir() + File.separator + "stardist";
		String scriptPath = envPath + File.separator + STARDIST3D_SCRIPT_NAME;
		
		GenericOp op = GenericOp.create(envPath, scriptPath, STARDIST3D_METHOD_NAME, 1);
		LinkedHashMap<String, Object> nMap = new LinkedHashMap<String, Object>();
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		nMap.put("input_" + dateString, image);
		nMap.put("nms_thresh", nms_threshold);
		nMap.put("prob_thresh", prob_threshold);
		op.setInputs(nMap);
		
		RunMode rm;
		rm = RunMode.createRunMode(op);
		Map<String, Object> resMap = rm.runOP();
		
		List<RandomAccessibleInterval<T>> rais = resMap.entrySet().stream()
				.filter(e -> {
					Object val = e.getValue();
					if (val instanceof RandomAccessibleInterval) return true;
					return false;
				}).map(e -> (RandomAccessibleInterval<T>) e.getValue()).collect(Collectors.toList());
		
		return rais.get(0);
	}
	
	public void checkRequirementsInstalled() {
		// TODO
	}
	
	public static void installRequirements() throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ArchiveException, URISyntaxException {
		boolean installed = InstalledEngines.buildEnginesFinder()
				.checkEngineWithArgsInstalledForOS("tensorflow", "1.15.0", null, null).size() != 0;
		if (!installed)
			EngineInstall.installEngineWithArgs("tensorflow", "1.15.0", true, true);
		
		Mamba mamba = new Mamba();
		boolean stardistPythonInstalled = false;
		try {
			stardistPythonInstalled = mamba.checkAllDependenciesInEnv("stardist", STARDIST_DEPS);
		} catch (MambaInstallException e) {
			mamba.installMicromamba();
		}
		if (!stardistPythonInstalled) {
			// TODO add logging for environment installation
			mamba.create("stardist", true, STARDIST_CHANNELS, STARDIST_DEPS);
		};
		String envPath = mamba.getEnvsDir() + File.separator + "stardist";
		String scriptPath = envPath + File.separator + STARDIST3D_SCRIPT_NAME;
		if (!Paths.get(scriptPath).toFile().isFile()) {
			try (InputStream scriptStream = Stardist3D.class.getClassLoader()
        			.getResourceAsStream(STARDIST3D_PATH_IN_RESOURCES + STARDIST3D_SCRIPT_NAME)){
    			Files.copy(scriptStream, Paths.get(scriptPath), StandardCopyOption.REPLACE_EXISTING);
    		}
		}
	}
	
	
	public static void main(String[] args) throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ModelSpecsException, LoadEngineException, 
													RunModelException, ArchiveException, 
													URISyntaxException, LoadModelException {
		Stardist3D.installRequirements();
		Stardist3D model = Stardist3D.fromPretained("StarDist Plant Nuclei 3D ResNet", false);
		
		RandomAccessibleInterval<FloatType> img = ArrayImgs.floats(new long[] {116, 120, 66});
		
		RandomAccessibleInterval<FloatType> res = model.predict(img);
		System.out.println(true);
	}
}
