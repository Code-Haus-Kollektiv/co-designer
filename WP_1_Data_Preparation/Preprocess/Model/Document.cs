using System;
using System.Collections.Generic;

namespace Preprocess.Model
{
    public class Document
    {
        public string Id { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public string FileName { get; set; }
        public string AuthorName { get; set; }
        public int AuthorLikes { get; set; }
        public int AuthorPostCount { get; set; }
        public string TopicName { get; set; }
        public int ObjectCount { get; set; }
        public DateTime CreatedAt { get; set; }
        public string PostUrl { get; set; }

        public List<Component> Components { get; set; }

    }
}