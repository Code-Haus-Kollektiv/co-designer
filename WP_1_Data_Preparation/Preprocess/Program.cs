using System;
using System.IO;
using System.Reflection;
using Grasshopper.Kernel;
using Newtonsoft.Json;
using System.Collections.Generic;
using Preprocess.Model;
using System.Linq;

namespace Preprocess
{
    class Program
    {

        static Program()
        {
            LoadLibraries();
            RhinoInside.Resolver.Initialize();
        }

        static void Main(string[] args)
        {
            var documents = ReadMetaDataCsv();
            documents.Reverse();

            using (var core = new Rhino.Runtime.InProcess.RhinoCore())
            {
                InitialiseGrasshopper();
                foreach (var document in documents)
                {
                    try
                    {
                        var ghDoc = LoadGrasshopperFile("./Files/downloaded_gh_files/" + document.FileName);
                        var components = new List<Component>();
                        components = GetComponentsFromDocument(ghDoc);

                        AddGrasshopperInfoToDocument(ghDoc, document);
                        document.Components = components;

                        WriteDocumentToJsonFile(document);

                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        continue;
                    }

                }


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
            var gh = Rhino.PlugIns.PlugIn.LoadPlugIn(new Guid("b45a29b1-4343-4035-989e-044e8580d9cf"), true, true);
            if (!gh) throw new Exception("Failed to load Grasshopper.");

        }

        /// <summary>
        /// Reads the CSV file that contains all the metadata of the grasshopper files.
        /// </summary>
        public static List<Document> ReadMetaDataCsv()
        {
            // Read the CSV file line by line
            string filePath = "./Files/files_metadata.csv";
            using (var reader = new StreamReader(filePath))
            {
                var headerLine = reader.ReadLine();
                var headers = headerLine.Split(',');

                // Read and process the data rows
                var allDocuments = new List<Document>();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    var newDoc = new Document()
                    {
                        FileName = values[0],
                        AuthorName = values[2],
                        AuthorLikes = 0, // Default to 0 if parsing fails
                        CreatedAt = DateTime.MinValue, // Default to MinValue if parsing fails
                        TopicName = values[6],
                        AuthorPostCount = 0, // Default to 0 if parsing fails
                        PostUrl = values[8],
                    };

                    // TryParse logic
                    if (int.TryParse(values[4], out int authorLikes))
                        newDoc.AuthorLikes = authorLikes;

                    if (DateTime.TryParse(values[5], out DateTime createdAt))
                        newDoc.CreatedAt = createdAt;

                    if (int.TryParse(values[7], out int authorPostCount))
                        newDoc.AuthorPostCount = authorPostCount;

                    allDocuments.Add(newDoc);

                    // Access data by column position
                    Console.WriteLine($"New document: {JsonConvert.SerializeObject(newDoc)}");
                }
                return allDocuments;
            }
        }

        /// <summary>
        /// Load Grasshopper file from filepath.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static GH_Document LoadGrasshopperFile(string filePath)
        {
            var document = new GH_DocumentIO();
            if (!document.Open(filePath))
            {
                throw new Exception("Could not load Grasshopper file with name ");
            }
            Console.WriteLine("GH Document with name: " + document.Document.DisplayName + "loaded");
            return document.Document;
        }


        /// <summary>
        /// Gets objects from Grasshopper canvas and returns an array of Component Objects.
        /// </summary>
        /// <param name="document"></param>
        /// <returns></returns>
        public static List<Component> GetComponentsFromDocument(GH_Document document)
        {
            Console.WriteLine("Document has " + document.ObjectCount + " objects.");
            List<Component> components = new List<Component>();
            foreach (var ghObject in document.Objects)
            {
                //Todo: include special objects like sliders/valuelists
                if (!(ghObject is IGH_Component componentObject)) { continue; }
                var obj = ghObject as IGH_Component;
                var component = new Component()
                {
                    Id = obj.ComponentGuid.ToString(),
                    InstanceId = obj.InstanceGuid.ToString(),
                    Type = typeof(IGH_Component),
                    Name = obj.Name,
                    Keywords = obj.Keywords == null ? new List<string>() : obj.Keywords.ToList(),
                    Description = obj.Description,
                    InstanceDescription = obj.InstanceDescription,
                    Nickname = obj.NickName,
                    Category = obj.Category,
                    SubCategory = obj.SubCategory,
                    Pivot = new Pivot(obj.Attributes.Pivot.X, obj.Attributes.Pivot.Y),
                    IsObsolete = obj.Obsolete,
                    IsLocked = obj.Locked,
                    Message = obj.Message,

                    Parameters = new List<Parameter>(),
                    DocumentInstanceId = document.DocumentID.ToString(),

                };

                var plugin = Grasshopper.Instances.ComponentServer.FindAssemblyByObject(obj);
                if (plugin != null)
                {
                    component.Plugin = new Plugin
                    {
                        Name = plugin.Name,
                        Id = plugin.Id.ToString(),
                        Version = plugin.Version,
                        Description = plugin.Description.ToString()
                    };
                }


                foreach (var input in componentObject.Params.Input)
                {
                    component.Parameters.Add(GetParameter(input, ParameterType.Input));
                }

                foreach (var output in componentObject.Params.Output)
                {
                    component.Parameters.Add(GetParameter(output, ParameterType.Output));
                }
                components.Add(component);
            }
            return components.ToList();
        }

        public static void WriteDocumentToJsonFile(Document document)
        {
            string json = JsonConvert.SerializeObject(document, Formatting.Indented);

            // Write JSON to file
            File.WriteAllText("./Results/" + document.FileName.Replace(".gh", ".json"), json);

            Console.WriteLine("Document " + document.FileName + "written successfuly.");

        }


        /// <summary>
        /// Gets a Parameter object from an IGH_Param Grasshopper object.
        /// </summary>
        /// <param name="param">The grasshopper IGH_Param object</param>
        /// <param name="parameterType">Input or Output</param>
        /// <returns></returns>
        public static Parameter GetParameter(IGH_Param param, ParameterType parameterType)
        {
            var parameter = new Parameter
            {
                //MetaData
                Id = param.ComponentGuid.ToString(),
                InstanceId = param.InstanceGuid.ToString(),
                ParameterType = parameterType,
                Name = param.Name,
                Nickname = param.NickName,
                Description = param.Description,

                //DataType
                DataType = param.DataType.GetType(),
                TypeName = param.TypeName,
                Access = (Access)param.Access,

                //State
                IsReversed = param.Reverse,
                IsSimplified = param.Simplify,
                IsLocked = param.Locked,
                IsNickNameMutable = param.MutableNickName,
                DataMapping = (DataMapping)param.DataMapping,
                WireDisplay = (WireDisplay)param.WireDisplay,

                //Volitale Data
                VolatileData = param.VolatileData,
                VolatileDataCount = param.VolatileDataCount,
            };

            var connectedParams = parameterType == ParameterType.Input ? param.Sources : param.Recipients;

            //Connected
            parameter.ConnectedCount = connectedParams.Count;
            if (connectedParams.Count > 0)
            {
                parameter.ConnectedIds = connectedParams.Select(o => o.ComponentGuid.ToString()).ToList();
                parameter.ConnectedInstanceIds = connectedParams.Select(o => o.InstanceGuid.ToString()).ToList();
            }
            ;
            return parameter;

        }


        public static void AddGrasshopperInfoToDocument(GH_Document ghDoc, Document document)
        {
            document.Id = ghDoc.DocumentID.ToString();
            document.Title = ghDoc.DisplayName;
            document.Description = "A json object of a Grasshopper file, containing JSON objects of all the " + ghDoc.ObjectCount + " IGH_Component.";
            document.ObjectCount = ghDoc.ObjectCount;
        }


    }
}