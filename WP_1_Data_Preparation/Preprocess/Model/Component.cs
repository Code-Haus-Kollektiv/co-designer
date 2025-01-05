using System.Collections.Generic;

namespace Preprocess.Model
{
    public class Component
    {
        // Metadata
        public string Id { get; set; }
        public string InstanceId { get; set; }
        public string Name { get; set; }
        public string Nickname { get; set; }
        public string Description { get; set; }
        public string Category { get; set; }
        public string SubCategory { get; set; }

        // Parameters
        public List<Parameter> Parameters { get; set; }

        // State
        public bool IsEnabled { get; set; }
        public bool IsHidden { get; set; }
        public (float X, float Y) Position { get; set; }

        //Document
        public string LibraryId { get; set; }
        public string DocumentInstanceId { get; set; }
    }
}