using System;
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
        public Type Type { get; set; }
        public List<string> Keywords { get; set; }
        public string Description { get; set; }
        public string InstanceDescription { get; set; }
        public string Category { get; set; }
        public string SubCategory { get; set; }
        public bool IsObsolete { get; set; }

        // Parameters
        public List<Parameter> Parameters { get; set; }

        // State
        public bool IsLocked { get; set; }
        public bool IsHidden { get; set; }
        public Pivot Pivot { get; set; }
        public string Message { get; set; }


        //Document
        public Plugin Plugin { get; set; }
        public string DocumentInstanceId { get; set; }
    }

    public class Plugin
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Version { get; set; }
        public string Description { get; set; }
    }

    public class Pivot
    {
        public Pivot(float X, float Y)
        {
            this.X = X;
            this.Y = Y;
        }

        public float X { get; set; }
        public float Y { get; set; }
    }
}