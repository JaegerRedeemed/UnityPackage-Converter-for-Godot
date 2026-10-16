# UnityPackage-Converter-for-Godot
Unity2Godot takes a .unitypackage file, such as one bought from the Unity Asset Store, and turns it into assets Godot 4 can use directly.

It unpacks the package and restores the original folder structure. It copies over textures, 3D models, and audio, and converts image formats Godot can’t read (.tif, .psd) to PNG. It rebuilds Unity materials as Godot materials, with all the textures already linked up, including baking Unity’s smoothness data into the roughness maps Godot expects. It also converts Unity terrain layers into tiling terrain materials and, where possible, exports terrain heightmaps as .r16 files.

You can use it through a simple window (pick the package and your Godot project folder, click Convert) or from the command line.

It doesn’t convert Unity-only things like prefabs, scenes, scripts, shaders, or painted terrain splatmaps. Those are skipped and listed in the log.


*made with Claude
