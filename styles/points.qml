<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<!-- test points: red cross; value-map form for label (1/0/9), conf (1/2) and issue. -->
<qgis version="3.34.0" styleCategories="Symbology|Fields|Forms">
  <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option type="QString" name="name" value="cross"/>
            <Option type="QString" name="color" value="255,0,0,255"/>
            <Option type="QString" name="outline_color" value="255,0,0,255"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0.5"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="size" value="5"/>
            <Option type="QString" name="size_unit" value="MM"/>
            <Option type="QString" name="angle" value="0"/>
            <Option type="QString" name="scale_method" value="diameter"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <fieldConfiguration>
    <field name="pid" configurationFlags="NoFlag">
      <editWidget type="TextEdit">
        <config><Option/></config>
      </editWidget>
    </field>
    <field name="label" configurationFlags="NoFlag">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="List" name="map">
              <Option type="Map"><Option type="QString" name="1 canopy" value="1"/></Option>
              <Option type="Map"><Option type="QString" name="0 not canopy" value="0"/></Option>
              <Option type="Map"><Option type="QString" name="9 cannot tell" value="9"/></Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="conf" configurationFlags="NoFlag">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="List" name="map">
              <Option type="Map"><Option type="QString" name="1 low" value="1"/></Option>
              <Option type="Map"><Option type="QString" name="2 high" value="2"/></Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="issue" configurationFlags="NoFlag">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option type="List" name="map">
              <Option type="Map"><Option type="QString" name="none" value="none"/></Option>
              <Option type="Map"><Option type="QString" name="shadow" value="shadow"/></Option>
              <Option type="Map"><Option type="QString" name="blur" value="blur"/></Option>
              <Option type="Map"><Option type="QString" name="leaf_off" value="leaf_off"/></Option>
              <Option type="Map"><Option type="QString" name="seam" value="seam"/></Option>
              <Option type="Map"><Option type="QString" name="edge" value="edge"/></Option>
              <Option type="Map"><Option type="QString" name="other" value="other"/></Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias field="pid" name="point"/>
    <alias field="label" name="label"/>
    <alias field="conf" name="confidence"/>
    <alias field="issue" name="issue"/>
  </aliases>
  <editable>
    <field name="pid" editable="0"/>
    <field name="label" editable="1"/>
    <field name="conf" editable="1"/>
    <field name="issue" editable="1"/>
  </editable>
  <editorlayout>generatedlayout</editorlayout>
</qgis>
