import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import React from "react";
import { AntDesign } from "@expo/vector-icons";
import { router } from 'expo-router'
import uuid from 'react-native-uuid'
interface tabHeaderTitle {
    addButton: boolean,
    backgroundColor: string,
    title: string;

}

const createChat = async (uuid: string) => {
    const response = await fetch('http://127.0.0.1:4000/api/v1/createChat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ uuid })
    })
    const { data } = await response.json()
    return data;
}

const generateUUID = () => {
    return uuid.v4()
}

export default function TabHeader({ title, addButton, backgroundColor }: tabHeaderTitle,) {
    return (
        <View style={{ ...styles.viewStyle, backgroundColor }}>
            <Text style={styles.textStyle}>{title}</Text>
            {addButton && <TouchableOpacity onPress={async () => {
                const uuid = generateUUID();
                const data = await createChat(uuid);
                router.push({ pathname: `./chats/${uuid}`, params: { chat_uuid: uuid } })
            }}>
                <AntDesign name='plus' size={30} />
            </TouchableOpacity>}
        </View>
    );
}

const styles = StyleSheet.create({
    viewStyle: {
        height: 70,
        justifyContent: "space-between",
        alignItems: 'center',
        padding: 10,
        display: "flex",
        flexDirection: 'row',
        borderBottomLeftRadius: 10,
        borderBottomRightRadius: 10
    },
    textStyle: {
        fontWeight: 'bold',
        fontSize: 20,
        textTransform: 'capitalize',
    }
});
