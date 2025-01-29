/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ModelDescriptorV05 extends ModelDescriptor
{

	protected ModelDescriptorV05(Map<String, Object> yamlElements)
    {
    	this.yamlElements = yamlElements;
    	buildModelDescription();
    }

	@Override
	public String getNickname() {
		return modelID;
	}

	@Override
	protected List<Author> buildAuthors(Object object) {
		List<Author> authors = new ArrayList<Author>();
    	if (object == null || !(object instanceof List)) {
            this.authors = authors;
            return authors;
    	}
        for (Object elem : (List<Object>) object)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, String> dict = (Map<String, String>) elem;
            authors.add(Author.build(dict.get("affiliation"), dict.get("email"), dict.get("github_user"), dict.get("name"), dict.get("orcid")));
        }
		return authors;
	}

	@Override
	protected List<Cite> buildCiteElements() {
		Object citeElements = this.yamlElements.get("cite");
        List<Cite> cites = new ArrayList<Cite>();
    	if (citeElements == null || !(citeElements instanceof List<?>)) {
    		this.cite = cites;
    		return cites;
    	}
        for (Object elem : (List) citeElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> dict = (Map<String, Object>) elem;
            cites.add(Cite.build((String) dict.get("text"), (String) dict.get("doi"), (String) dict.get("url")));
        }
		return cites;
	}

	@Override
	protected List<TensorSpec> buildInputTensors() {
		Object object = this.yamlElements.get("inputs");
		if (!(object instanceof List<?>))
    		return null;
		List<?> list = (List<?>) object;
        List<TensorSpec> tensors = new ArrayList<>(list.size());
        for (Object elem : list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(new TensorSpecV05((Map<String, Object>) elem, true));
        }
        return tensors;
	}

	@Override
	protected List<TensorSpec> buildOutputTensors() {
		Object object = this.yamlElements.get("outputs");
		if (!(object instanceof List<?>))
    		return null;
		List<?> list = (List<?>) object;
        List<TensorSpec> tensors = new ArrayList<>(list.size());
        for (Object elem : list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(new TensorSpecV05((Map<String, Object>) elem, false));
        }
        return tensors;
	}

	@Override
	protected void calculateTotalInputHalo() {
		for (TensorSpec out: output_tensors) {
			for (Axis ax : out.getAxesInfo().getAxesList()) {
				int axHalo = ax.getHalo();
				if (axHalo == 0)
					continue;
				String ref = ax.getReferenceTensor();
				if (ref == null) {
					this.input_tensors.stream().forEach( tt -> {
						AxisV05 inAx = (AxisV05) tt.getAxesInfo().getAxesList().stream()
						.filter(xx -> xx.getAxis().equals(ax.getAxis()))
						.findFirst().orElse(null);
						if (inAx == null || inAx.getHalo() > axHalo) return;
						inAx.halo = axHalo;
					});
					return;
				}
				
				double axScale = ax.getScale();
				double axOffset = ax.getOffset();
				double nHalo = (axHalo + axOffset) / axScale;
				AxisV05 inAx = (AxisV05) this.findInputTensor(ref).getAxesInfo().getAxis(ax.getReferenceAxis());

				if (inAx == null || inAx.getHalo() > nHalo) return;
				inAx.halo = (int) nHalo;
			}
		}
		
	}

	@Override
	protected ExecutionConfig buildConfig() {
        return ExecutionConfig.build((Map<String, Object>) this.yamlElements.get("config"));
	}

	@Override
	protected ModelWeight buildWeights() {
        return ModelWeight.build((Map<String, Object>) this.yamlElements.get("weights"));
	}

	@Override
	protected String findID() {
		if (yamlElements.get("config") != null && yamlElements.get("config") instanceof Map) {
    		Map<String, Object> configMap = (Map<String, Object>) yamlElements.get("config");
    		if (configMap.get("bioimageio") != null && configMap.get("bioimageio") instanceof Map) {
    			Map<String, Object> bioimageMap = (Map<String, Object>) configMap.get("bioimageio");
    			if (bioimageMap.get("nickname") != null)
    				return (String) bioimageMap.get("nickname");
    		}
    	}
    	return (String) yamlElements.get("id");
	}

	@Override
	// TODO decide whether to keep this or not
	protected void addBioEngine() {
		// TODO decide what to do with servers. Probably need permissions / Implement authentication
    	if (getName().equals("cellpose-python")) {
    		supportBioengine = true;
			return;
	    } else if (getName().equals("bestfitting-inceptionv3-single-cell")) {
			return;
	    } else if (getName().equals("stardist")) {
    		supportBioengine = true;
			return;
	    } else if (modelID == null) {
    		supportBioengine = false;
	    	return;
	    }
    	supportBioengine =  BioimageioRepo.isModelOnTheBioengineById(modelID);
		
	}
}
