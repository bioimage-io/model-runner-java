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
package io.bioimage.modelrunner.versionmanagement;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;

/**
 * Class to create an object that contains all the information about a Deep
 * Learning framework (engine) that is needed to launch the engine in an
 * independent ClassLoader
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class SupportedVersions
{
	/**
	 * Map containing all the versions supported by each framework and their 
	 * correspondence between Java and Python
	 */
	private static HashMap< String, Object > ALL_VERSIONS;

	/**
	 * Key for the Java equivalent version in the JSON file
	 */
	private static String javaVersionsKey = "javaVersion";

	/**
	 * HashMap containing all the versions supported for a specific engine and
	 * their corresponding Java versions
	 */
	private LinkedTreeMap< String, Object > versionsDic;

	/**
	 * Set of Strings where each entry corresponds to a supported Deep Learning
	 * framework version
	 */
	private Set< String > versionSet;
	
	/**
	 * Method to test which are the versions that should ver returned in every 
	 * case
	 * @param args
	 * 	not used
	 */
	public static void main(String[] args) {
		String vv = getClosestSupportedPythonVersion("tensorflow", "2.13.0");
		String v1 = getClosestSupportedPythonVersion("tensorflow", "2");
		String v2 = getClosestSupportedPythonVersion("tensorflow", "2.8");
		String v3 = getClosestSupportedPythonVersion("tensorflow", "2.1");
		String v4 = getClosestSupportedPythonVersion("tensorflow", "2.1.70");
		String v5 = getClosestSupportedPythonVersion("tensorflow", "3");
		String v7 = getClosestSupportedPythonVersion("onnx", "20");
		String v8 = getClosestSupportedPythonVersion("onnx", "13");
		System.out.print(false);
	}

	/**
	 * Class to find the version of Deep Learning framework (engine) equivalent
	 * or compatible with the one used to train the model. This is done because
	 * sometimes APIs for different languages are named differently
	 * 
	 * @param engine
	 *            Deep Learning framework which is going to be loaded
	 */
	public SupportedVersions( String engine )
	{
		engine = AvailableEngines.getSupportedFrameworkTag(engine);
    	if (engine == null) 
    		this.versionsDic = new LinkedTreeMap<String, Object>();
    	else
    		this.versionsDic = getSupportedVersionsForEngine( engine );
    	if (versionsDic == null)
    		this.versionSet = Collections.<String>emptySet();
    	else
    		this.versionSet = this.versionsDic.keySet();
	}
	
	/**
	 * Finds the closest supported version by JDLL for the wanted
	 * Deep Learning framework version wanted
	 * @param version
	 * 	version in Python of the Deep Learning framework selected
	 * @return the version in Python closest to the provided one for 
	 * 	the Deep LEarning framework selected
	 */
	public String getClosestSupportedPythonVersion(String version) {
		return findVersionInJSON( version, versionSet );
	}

	/**
	 * Find the corresponding version of the API to run a Deep Learning
	 * framework in Java.
	 * 
	 * @param version
	 *            version of the Deep Learning framework (engine) used to create
	 *            the model
	 * @return the corresponding Java version
	 */
	public String getCorrespondingJavaVersion( String version )
	{
		version = findVersionInJSON( version, versionSet );
		// TODO warn that the version used is not the exact same one as the
		// one created
		return getJavaVersionFromVersionJSON( version, this.versionsDic );
	}

	/**
	 * Obtain from the plugin resources json the currently supported Deep
	 * Learning frameworks and versions
	 * 
	 * @return a hashmap containing the json file with the all supported
	 *         versions
	 */
	public static HashMap< String, Object > readVersionsJson()
	{
		BufferedReader br = new BufferedReader( new InputStreamReader(
				SupportedVersions.class.getClassLoader().getResourceAsStream( "supportedVersions.json" ) ) );
		Gson g = new Gson();
		// Create the type that we want to read from the json file
		Type mapType = new TypeToken< HashMap< String, Object > >(){}.getType();
		HashMap< String, Object > supportedVersions = g.fromJson( br, mapType );
		return supportedVersions;
	}

	/**
	 * Get the supported versions for an specific Deep Learning framework
	 * (engine)
	 * 
	 * @param engine
	 *            the Deep Learning framework we want
	 * @return a HashMap containing all the supported versions for a Deep
	 *         Learning framework
	 */
	public static LinkedTreeMap< String, Object > getSupportedVersionsForEngine( String engine )
	{
		if (ALL_VERSIONS == null)
			ALL_VERSIONS = readVersionsJson();
		engine = AvailableEngines.getSupportedFrameworkTag(engine);
		LinkedTreeMap< String, Object > engineVersions = ( LinkedTreeMap< String, Object > ) ALL_VERSIONS.get( engine );
		return engineVersions;
	}

	/**
	 * Get the closest Deep Learning framework (engine) version to the one
	 * provided. If no version coincides exactly with the ones allowed, retrieve
	 * the closest one. For example if there is no version 2.7.1, retrieve
	 * 2.7.5. In the worst case scenario However, at least the major version has
	 * to coincide. For example version 2.1.0 will not be never retrieved if the
	 * wanted version is 1.2.2. In the case there is no version 1.X.Y an
	 * exception will be thrown.
	 * 
	 * @param version
	 *            The wanted version of the Deep Learning framework
	 * @param versionSet
	 *            Set of all the versions supported by the program
	 * @return the closest version to the available one, return
	 *  null if there is no compatible version
	 */
	private static String findVersionInJSON( String version, Set< String > versionSet )
	{
		// Get the version with only major and minor version numbers, no
		// revision number
		// For example 2.8.1 -> 2.8. If the version already has not the revision
		// number
		// leave it as it is.
		if (versionSet.contains(version))
			return version;
		if ( version.indexOf( "." ) != -1 && version.indexOf( "." ) != version.lastIndexOf( "." ) )
		{
			int secondDotPos = version.substring( version.indexOf( "." ) + 1).indexOf( "." );
			version = version.substring( 0, version.indexOf( "." ) + 1 + secondDotPos );
		}
		List< String > auxVersionList = versionSet.stream().map( s -> {
			if (s.indexOf(".") == -1 || s.indexOf(".") == s.lastIndexOf("."))
				return s;
			return s.substring( 0, s.lastIndexOf( "." ));
			}).collect( Collectors.toList() );
		if ( auxVersionList.contains( version ) )
			return ( String ) versionSet.toArray()[ auxVersionList.indexOf( version ) ];
		// If there is still no coincidence, just look for the major version.
		// For example, in 2.3.4 just look for the most recent 2 version
		if ( version.indexOf( "." ) != -1 )
			version = version.substring( 0, version.indexOf( "." ) );
		auxVersionList = auxVersionList.stream().map( s -> {
			if (s.indexOf(".") == -1)
				return s;
			return s.substring( 0, s.indexOf( "." ));
			}).collect( Collectors.toList() );
		if ( auxVersionList.contains( version ) )
			return ( String ) versionSet.toArray()[ auxVersionList.indexOf( version ) ];
		return null;
	}

	/**
	 * Retrieve the JavaVersion key from the SupportedVErsions HashMap read from
	 * the JSON
	 * 
	 * @param version
	 *            version of the Deep Learning framework as it is written in the
	 *            JSON file. 
	 * @param allVersions
	 *            list of all supported versions
	 * @return get the Java version for a specific Pyrhon version
	 */
	private static String getJavaVersionFromVersionJSON( String version, LinkedTreeMap< String, Object > allVersions )
	{
		if (version == null)
			return null;
		LinkedTreeMap< String, String > versionJSON = ( LinkedTreeMap< String, String > ) allVersions.get( version );
		return versionJSON.get( javaVersionsKey );
	}

	/**
	 * Get the supported versions for the wanted Deep Learning framework
	 * (engine)
	 * 
	 * @return set of supported versions
	 */
	public Set< String > getSupportedVersions()
	{
		return this.versionSet;
	}
	
	/**
	 * Find the Java engine version that is compatible with the Python engine version provided
	 * @param engine
	 * 	the engine of interest
	 * @param version
	 * 	the python version of interest
	 * @return the python version of interest or null if it does not exist
	 */
	public static String getJavaVersionForPythonVersion(String engine, String version) {
		SupportedVersions sv = new SupportedVersions(engine);
		return sv.getCorrespondingJavaVersion(version);
	}
	
	/**
	 * Finds the closest supported version by JDLL for the wanted
	 * Deep Learning framework version wanted
	 * @param engine
	 * 	Deep LEarning frameework of interest
	 * @param version
	 * 	version in Python of the Deep Learning framework selected
	 * @return the version in Python closest to the provided one for 
	 * 	the Deep LEarning framework selected
	 */
	public static String getClosestSupportedPythonVersion(String engine, String version) {
		SupportedVersions sv = new SupportedVersions(engine);
		return sv.getClosestSupportedPythonVersion(version);
	}
}
