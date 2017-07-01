import xml.etree.ElementTree as ET
import struct
import base64
import re
import numpy as np

class VTKXMLParser(ET.XMLParser):
    raw_start_pattern = re.compile(r"<AppendedData encoding=\"raw\">_")
    raw_end_pattern = re.compile(r"</AppendedData>")
    raw_sub_pattern = re.compile(r"(?<=<AppendedData encoding=\"raw\">)\s*_(.*)\s*(?=</AppendedData>)")

    def __init__(self, *args, **kwargs):
        if not "target" in kwargs:
            kwargs["target"] = VTKTreeBuilder()

        self.incomplete_raw_tag = False
        self.raw_data = ""

        super(VTKXMLParser, self).__init__(*args, **kwargs)

    def feed(self, data, *args, **kwargs):
        """Do some Cthulhu parsing instead of passing raw binary data
        """
        if self.raw_start_pattern.search(data) and not self.raw_end_pattern.search(data):
            self.incomplete_raw_tag = True
        if self.incomplete_raw_tag:
            self.raw_data += data
            if self.raw_sub_pattern.search(self.raw_data):
                self.incomplete_raw_tag = False
                data = self.raw_data
                self.raw_data = ""
            else:
                return
        data = self.raw_sub_pattern.sub(lambda x: base64.b64encode(x.group(1)), data)
        super(VTKXMLParser, self).feed(data, *args, **kwargs)

class VTKTreeBuilder(ET.TreeBuilder):
    """Decodes XML-based VTK files and exports all array data to NumPy.

    VTK specification found at http://www.vtk.org/Wiki/VTK_XML_Formats
    """

    # Buffer codes from https://docs.python.org/2/library/struct.html
    buffers = {"Int8": "b", "UInt8": "B", "Int16": "h", "UInt16": "H",
               "Int32": "i", "UInt32": "I", "Int64": "q", "UInt64": "Q",
               "Float32": "f", "Float64": "d"}
    python_types = {"Int8": int, "UInt8": int, "Int16": int, "UInt16": int,
                    "Int32": int, "UInt32": int, "Int64": int, "UInt64": int,
                    "Float32": float, "Float64": float}
    byteorders = {"LittleEndian": "<", "BigEndian": ">"}

    def __init__(self, *args, **kwargs):
        self.appended_data_arrays = []
        super(VTKTreeBuilder, self).__init__(*args, **kwargs)


    def start(self, tag, attrib):
        self.elem = super(VTKTreeBuilder,self).start(tag, attrib)
        self.array_data = ""
        if tag == "VTKFile":
            try:
                self.split_header = attrib["version"] == "0.1"
            except KeyError:
                raise ValueError("Missing version attribute in VTKFile tag")
            try:
                self.byteorder = self.byteorders[attrib["byte_order"]]
            except KeyError:
                raise ValueError("Unknown byteorder {}".format(attrib["byte_order"]))
            try:
                self.header_type = self.buffers[attrib["header_type"]]
            except KeyError: # default header in older VTK versions is UInt32
                self.header_type = "I"
        elif tag == "DataArray":
            if attrib["format"] == "appended":
                self.appended_data_arrays.append((int(attrib["offset"]), attrib["type"], self.elem))
            if attrib["format"] not in ("binary","ascii","appended"):
                raise ValueError("VTK data format must be 'ascii', 'binary' (base64), or 'appended'. Got: {}".format(attrib["format"]))
        return self.elem


    def data(self, data):
        """Just record the data instead of writing it immediately. All data in VTK
        files is contained in DataArray tags, so no need to record anything if
        we are not currently in a DataArray.
        """
        if self.elem.tag in {"DataArray", "AppendedData"}:
            self.array_data += data


    def end(self, tag):
        """Detect the end-tag of a DataArray or AppendedData element.
        """
        if tag == "DataArray":
            self.handle_data_array()
        elif tag == "AppendedData":
            self.handle_appended_data()
        self.array_data = ""
        return super(VTKTreeBuilder, self).end(tag)


    def handle_data_array(self):
        cbuf = self.byteorder + self.buffers[self.elem.attrib["type"]]
        if self.elem.attrib["format"] == "binary":
            input_data = self.array_data.strip()
            # binary encoded VTK files start with an integer giving the number of bytes to follow
            header_size = len(base64.b64encode(struct.pack(self.byteorder + self.header_type, 0)))
            data_len = struct.unpack_from(self.byteorder + self.header_type, base64.b64decode(input_data[:header_size]))[0]
            if self.split_header: # vtk version 0.1, encoding header and content separately
                data_content = base64.b64decode(input_data[header_size:])
                byte_string = self.byteorder + cbuf * int(data_len / struct.calcsize(cbuf))
                data_bytelen = int(data_len / struct.calcsize(cbuf))
                array_data = np.fromstring(data_content, dtype=cbuf, count=data_bytelen)
            else: # vtk version 1.0, encoding header and content together
                data_content = base64.b64decode(array_data)
                data_bytelen = int(data_len / struct.calcsize(cbuf))
                header_size = struct.calcsize(self.byteorder + self.header_type)
                array_data = np.fromstring(data_content[header_size:], dtype=cbuf, count=data_bytelen)
        else:
            array_data = np.fromstring(self.array_data, dtype=cbuf, sep=' ')
        ncomp = int(self.elem.attrib.get("NumberOfComponents") or 0)
        if ncomp > 1:
            array_data = array_data.reshape(-1, int(ncomp))
        self.elem.text = array_data


    def handle_appended_data(self):
        """Decodes base64 encoded appended data and adds it to respective parent element
        """
        raw_data = base64.b64decode(self.array_data)
        header_bytestring = self.byteorder + self.header_type
        header_size = struct.calcsize(header_bytestring)
        for offset, dtype, parent_element in self.appended_data_arrays:
            datalen = struct.unpack_from(header_bytestring, raw_data, offset=offset)[0]
            cbuf = self.buffers[dtype]
            data_bytelen = int(datalen / struct.calcsize(cbuf))
            parent_element.text = np.fromstring(raw_data[offset + header_size:], dtype=cbuf, count=data_bytelen)


if __name__ == "__main__":
    import sys
    data = ET.parse(sys.argv[1],parser=VTKXMLParser())
    for child in data.iter():
        print(child.tag, child.text)
