using System;
using System.IO;
using System.Reflection;
using Grasshopper.Kernel;
using Newtonsoft.Json;
using Rhino.Geometry;
using System.Collections.Generic;
using Preprocess.Model;

namespace Preprocess
{
    class Program
    {
        static Program()
        {
            RhinoInside.Resolver.Initialize();
        }

        static void Main(string[] args)
        {
            LoadLibraries();

            using (var core = new Rhino.Runtime.InProcess.RhinoCore())
            {
                InitialiseGrasshopper();
            }
        }

        /// <summary>
        /// Dynamically adds RhinoCommon and Grasshopper dependencies to project.(assuming Rhino 8)
        /// </summary>
        private static void LoadLibraries()
        {
            string rhinoSystemDir = @"C:\Program Files\Rhino 8\System"; // Adjust for Rhino 8 if needed
            string grasshopperDir = Path.Combine(rhinoSystemDir, @"..\Plug-ins\Grasshopper");

            // Add Rhino system directory to the assembly resolver
            AppDomain.CurrentDomain.AssemblyResolve += (sender, eventArgs) =>
            {
                string assemblyName = new AssemblyName(eventArgs.Name).Name + ".dll";
                string assemblyPath = Path.Combine(rhinoSystemDir, assemblyName);

                if (!File.Exists(assemblyPath))
                    assemblyPath = Path.Combine(grasshopperDir, assemblyName);

                return File.Exists(assemblyPath) ? Assembly.LoadFrom(assemblyPath) : null;
            };

        }

        /// <summary>
        /// Initializes Grasshopper when Rhino has started.
        /// </summary>
        /// <exception cref="Exception"></exception>
        static void InitialiseGrasshopper()
        {
            // Start grasshopper in "headless" mode
            var gh = Rhino.PlugIns.PlugIn.LoadPlugIn(new Guid("b45a29b1-4343-4035-989e-044e8580d9cf"));
            if (!gh) throw new Exception("Failed to load Grasshopper.");
        }

        /// <summary>
        /// Load Grasshopper file from filepath.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static GH_Document LoadGrasshopperFile(string filePath)
        {
            var io = new GH_DocumentIO();

            if (!io.Open(filePath))
            {
                throw new Exception("Could not load Grasshopper file with name ");
            }
            Console.WriteLine("GH Document with name: " + io.Document.DisplayName + "loaded");
            return io.Document;
        }

        public static Component[] GetComponentsFromDocument(GH_Document document)
        {
            List<Component> components = new List<Component>();
            foreach (var obj in document.Objects)
            {
                var component = new Component();
                component.Id = obj.ComponentGuid.ToString();
                component.InstanceId = obj.InstanceGuid.ToString();
                component.Name = obj.Name;
                component.Description = obj.Description;
                component.Nickname = obj.NickName;
                component.Category = obj.Category;
                component.SubCategory = obj.SubCategory;
                component.Position = (obj.Attributes.Pivot.X, obj.Attributes.Pivot.Y);
                components.Add(component);
            }
            return components.ToArray();
        }


    }
}