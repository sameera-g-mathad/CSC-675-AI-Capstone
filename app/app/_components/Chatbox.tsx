import React, { useState } from "react";
import { TouchableOpacity, StyleSheet, Text, TextInput, View } from "react-native";
import { MaterialIcons } from "@expo/vector-icons";

export default function Chatbox() {
    const [query, setQuery] = useState("");
    return (
        <View style={styles.viewStyles}>
            <Text>{query}</Text>
            <TextInput value={query} onChangeText={setQuery} style={styles.inputStyle} placeholder="How does a historian evaluate events?" placeholderTextColor='#94a3b8' />
            <TouchableOpacity
                onPress={() => {
                    setQuery('');
                }
                }
                style={
                    {
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        width: '15%'
                    }}>
                <MaterialIcons name='arrow-forward' color='#475569' size={30} />
            </TouchableOpacity>
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyles: {
        height: 60,
        display: 'flex',
        flexDirection: 'row',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        backgroundColor: 'white'
    },
    inputStyle: {
        flex: 1,
        paddingLeft: 10,
        outlineWidth: 0,
    }
});
