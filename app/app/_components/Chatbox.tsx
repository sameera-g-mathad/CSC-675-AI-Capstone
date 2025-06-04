import React, { useContext, useState } from "react";
import { TouchableOpacity, StyleSheet, Text, TextInput, View } from "react-native";
import { MaterialIcons } from "@expo/vector-icons";
import ChatHistContext from "../_context/ChathistContext";

interface chatBoxInterface {
    uuid: string
}
export default function Chatbox({ uuid }: chatBoxInterface) {
    const [query, setQuery] = useState("");
    const { askQuery } = useContext(ChatHistContext)
    return (
        <View style={styles.viewStyles}>
            <TextInput value={query} multiline={true} onChangeText={setQuery} style={styles.inputStyle} placeholder="How does a historian evaluate events?" placeholderTextColor='#94a3b8' />
            <TouchableOpacity
                onPress={() => {
                    askQuery(uuid, query)
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
        alignItems: 'center',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        backgroundColor: 'white'
    },
    inputStyle: {
        flex: 1,
        padding: 10,
        lineHeight: 20,
        letterSpacing: 0.5,
        outlineWidth: 0,
    }
});
