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
package io.bioimage.modelrunner.models;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class Models {
	
	private static final List<String> STARDIST_DEPS = Arrays.asList(new String[] {"python=3.10", "stardist", "numpy"});
	
	private static final String STARDIST2D_PATH_IN_RESOURCES = "ops/stardist_postprocessing/";
	
	private static final String STARDIST2D_SCRIPT_NAME= "stardist_postprocessing.py";
	
	public static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> stardist(String modelPath, RandomAccessibleInterval<T> image, boolean returnPolys, boolean install) 
			throws IOException, InterruptedException, 
			IllegalStateException, LoadEngineException, 
			RunModelException {
		
		boolean installed = InstalledEngines.buildEnginesFinder()
				.checkEngineWithArgsInstalledForOS("tensorflow", "1.15.0", null, null).size() != 0;
		if (!installed && !install)
			throw new IllegalArgumentException("Tensorflow is required but it is not installed. "
					+ "Install it manually or set argument 'install=true'");
		else if (!installed)
			EngineInstall.installEngineWithArgs("tensorflow", "1.15", true, true);
		
		EngineInfo tfJavaEngine  = EngineInfo.defineDLEngine("tensorflow", "1.15");
		Model model = Model.createDeepLearningModel(modelPath, modelPath, tfJavaEngine);
		
		Tensor<T> inputTensor = Tensor.build(modelPath, modelPath, image);
		
		List<Tensor<T>> inputs = new ArrayList<Tensor<T>>();
		model.runModel(null, null);
		
		Mamba mamba = new Mamba();
		
		boolean stardistPythonInstalled = false;
		try {
			stardistPythonInstalled = mamba.checkAllDependenciesInEnv("stardist", STARDIST_DEPS);
		} catch (MambaInstallException e) {
		}
		if (!stardistPythonInstalled)
			throw new IllegalArgumentException();
		
		String envPath = mamba.getEnvsDir() + File.separator + "stardist";
		String scriptPath = envPath + File.separator + STARDIST2D_SCRIPT_NAME;
		if (!Paths.get(scriptPath).toFile().isFile()) {
			
		}
		
		
		return null;
	}
	
	public static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> stardist(String modelPath, RandomAccessibleInterval<T> image, boolean returnPolys) {
		
	}
}
